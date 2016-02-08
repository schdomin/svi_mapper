#include <opencv/highgui.h>
#include <stack>
#include <thread>

//ds custom
#include "txt_io/message_reader.h"
#include "../core/CTrackerSV.h"
#include "exceptions/CExceptionLogfileTree.h"
#include "utility/CParameterBase.h"
#include "optimization/Cg2oOptimizer.h"



//ds command line parsing (setting params IN/OUT)
void setParametersNaive( const int& p_iArgc,
                         char** const p_pArgv,
                         std::string& p_strMode,
                         std::string& p_strInfileCameraIMUMessages );

void printHelp( );



int32_t main( int32_t argc, char **argv )
{
    //assert( false );

    //ds defaults
    std::string strMode              = "benchmark";
    std::string strInfileMessageDump = "";
    std::string strConfigurationCameraLEFT  = "../hardware_parameters/kitti_00_camera_left.txt";
    std::string strConfigurationCameraRIGHT = "../hardware_parameters/kitti_00_camera_right.txt";

    //ds get params
    setParametersNaive( argc, argv, strMode, strInfileMessageDump );

    //ds escape here on failure
    if( strInfileMessageDump.empty( ) )
    {
        std::printf( "[0](main) no message file specified\n" );
        printHelp( );
        return 1;
    }

    //ds get playback mode to enum
    EPlaybackMode eMode( ePlaybackInteractive );

    if( "stepwise" == strMode )
    {
        eMode = ePlaybackStepwise;
    }
    else if( "benchmark" == strMode )
    {
        eMode = ePlaybackBenchmark;
    }

    //ds internals
    uint32_t uWaitKeyTimeout( 1 );

    //ds adjust depending on mode
    switch( eMode )
    {
        case ePlaybackStepwise:
        {
            uWaitKeyTimeout = 0;
            break;
        }
        case ePlaybackBenchmark:
        {
            uWaitKeyTimeout = 1;
            break;
        }
        default:
        {
            //ds exit
            std::printf( "[0](main) interactive mode not supported, aborting\n" );
            printHelp( );
            return 0;
        }
    }

    //ds message loop
    txt_io::MessageReader cMessageReader;
    cMessageReader.open( strInfileMessageDump );

    //ds escape here on failure
    if( !cMessageReader.good( ) )
    {
        std::printf( "[0](main) unable to open message file: '%s'\n", strInfileMessageDump.c_str( ) );
        printHelp( );
        return 1;
    }

    /*ds allocated loggers
    CLogger::CLogDetectionEpipolar::open( );
    CLogger::CLogLandmarkCreation::open( );
    CLogger::CLogLandmarkFinal::open( );
    CLogger::CLogLandmarkFinalOptimized::open( );
    CLogger::CLogOptimizationOdometry::open( );
    CLogger::CLogTrajectory::open( );
    CLogger::CLogIMUInput::open( );*/

    //ds log configuration
    CLogger::openBox( );
    std::printf( "[0](main) strConfigurationCameraLEFT  := '%s'\n", strConfigurationCameraLEFT.c_str( ) );
    std::printf( "[0](main) strConfigurationCameraRIGHT := '%s'\n", strConfigurationCameraRIGHT.c_str( ) );
    std::printf( "[0](main) strInfileMessageDump        := '%s'\n", strInfileMessageDump.c_str( ) );
    //std::printf( "(main) openCV build information: \n%s", cv::getBuildInformation( ).c_str( ) );
    CLogger::closeBox( );

    try
    {
        //ds load camera parameters
        CParameterBase::loadCameraLEFT( strConfigurationCameraLEFT );
        std::printf( "[0](main) successfully imported camera LEFT\n" );
        CParameterBase::loadCameraRIGHT( strConfigurationCameraRIGHT );
        std::printf( "[0](main) successfully imported camera RIGHT\n" );
        CParameterBase::constructCameraSTEREO( Eigen::Vector3d( -0.54, 0.0, 0.0 ) );
    }
    catch( const CExceptionParameter& p_cException )
    {
        std::printf( "[0](main) unable to import camera parameters - CExceptionParameter: '%s'\n", p_cException.what( ) );
        return 1;
    }
    catch( const std::invalid_argument& p_cException )
    {
        std::printf( "[0](main) unable to import camera parameters - std::invalid_argument: '%s'\n", p_cException.what( ) );
        return 1;
    }
    catch( const std::out_of_range& p_cException )
    {
        std::printf( "[0](main) unable to import camera parameters - std::out_of_range: '%s'\n", p_cException.what( ) );
        return 1;
    }

    //ds stop time
    const double dTimeStartSeconds = CLogger::getTimeSeconds( );

    //ds message holders
    std::shared_ptr< txt_io::PinholeImageMessage > pMessageCameraLEFT( 0 );
    std::shared_ptr< txt_io::PinholeImageMessage > pMessageCameraRIGHT( 0 );

    //ds must be calibrated
    assert( 0 != CParameterBase::pCameraLEFT );
    assert( 0 != CParameterBase::pCameraRIGHT );
    assert( 0 != CParameterBase::pCameraSTEREO );

    //ds allocate the tracker
    CTrackerSV cTracker( CParameterBase::pCameraSTEREO,
                         eMode,
                         uWaitKeyTimeout );
    try
    {
        //ds prepare file structure
        cTracker.sanitizeFiletree( );
    }
    catch( const CExceptionLogfileTree& p_cException )
    {
        std::printf( "[0](main) unable to sanitize file tree - exception: '%s'\n", p_cException.what( ) );
        printHelp( );
        return 1;
    }

    //ds count invalid frames
    UIDFrame uInvalidFrames = 0;

    //ds playback the dump
    while( cMessageReader.good( ) && !cTracker.isShutdownRequested( ) )
    {
        //ds retrieve a message
        txt_io::BaseMessage* msgBase = cMessageReader.readMessage( );

        //ds if set
        if( 0 != msgBase )
        {
            //ds camera message
            std::shared_ptr< txt_io::PinholeImageMessage > pMessageImage( dynamic_cast< txt_io::PinholeImageMessage* >( msgBase ) );
            assert( 0 != pMessageImage );

            //ds if its the left camera
            if( "camera_left" == pMessageImage->frameId( ) )
            {
                pMessageCameraLEFT  = pMessageImage;
            }
            else
            {
                pMessageCameraRIGHT = pMessageImage;
            }
        }

        //ds as soon as we have data in all the stacks - process
        if( 0 != pMessageCameraLEFT && 0 != pMessageCameraRIGHT )
        {
            //ds if the timestamps match (optimally the case)
            if( pMessageCameraLEFT->timestamp( ) == pMessageCameraRIGHT->timestamp( ) )
            {
                //ds evaluate images and IMU (inner landmark locking)
                cTracker.process( pMessageCameraLEFT, pMessageCameraRIGHT );

                //ds reset holders
                pMessageCameraLEFT.reset( );
                pMessageCameraRIGHT.reset( );

                //ds check reset
                assert( 0 == pMessageCameraLEFT );
                assert( 0 == pMessageCameraRIGHT );
            }
            else
            {
                //ds check timestamp mismatch
                if( pMessageCameraLEFT->timestamp( ) < pMessageCameraRIGHT->timestamp( ) )
                {
                    std::printf( "[0](main) timestamp mismatch LEFT: %f < RIGHT: %f - processing skipped\n", pMessageCameraLEFT->timestamp( ), pMessageCameraRIGHT->timestamp( ) );
                    pMessageCameraLEFT.reset( );
                }
                else
                {
                    std::printf( "[0](main) timestamp mismatch LEFT: %f > RIGHT: %f - processing skipped\n", pMessageCameraLEFT->timestamp( ), pMessageCameraRIGHT->timestamp( ) );
                    pMessageCameraRIGHT.reset( );
                }

                ++uInvalidFrames;
            }
        }
    }

    //ds get end time
    const double dDurationTotal  = CLogger::getTimeSeconds( )-dTimeStartSeconds;
    const double dDurationG2o    = cTracker.getTotalDurationOptimizationSeconds( );
    const double dDurationPure   = dDurationTotal-dDurationG2o;
    const UIDFrame uFrameCount   = cTracker.getFrameCount( );
    const double dDurationRealTime = uFrameCount/20.0;
    const double dDistance       = cTracker.getDistanceTraveled( );

    //ds if we processed data
    if( 1 < uFrameCount )
    {
        //ds finalize tracker (e.g. do a last optimization)
        cTracker.finalize( );

        //ds summary
        CLogger::openBox( );
        std::printf( "[0](main) dataset completed\n" );

        std::printf( "\n[0](main) frame rate (avg): %f fps (%4.2fx real time)\n", uFrameCount/dDurationPure, dDurationRealTime/dDurationTotal );

        std::printf( "\n[0](main) duration             Total: %7.2fs (1.00) Real: %7.2fs\n", dDurationTotal, dDurationRealTime );
        std::printf( "[0](main) duration Regional Tracking: %7.2fs (%4.2f)\n", cTracker.getDurationTotalSecondsRegionalTracking( ), cTracker.getDurationTotalSecondsRegionalTracking( )/dDurationTotal );
        std::printf( "[0](main) duration Epipolar Tracking: %7.2fs (%4.2f)\n", cTracker.getDurationTotalSecondsEpipolarTracking( ), cTracker.getDurationTotalSecondsEpipolarTracking( )/dDurationTotal );
        std::printf( "[0](main) duration       StereoPosit: %7.2fs (%4.2f)\n", cTracker.getDurationTotalSecondsStereoPosit( ), cTracker.getDurationTotalSecondsStereoPosit( )/dDurationTotal );
        std::printf( "[0](main) duration      Loop closing: %7.2fs (%4.2f)\n", cTracker.getDurationTotalSecondsLoopClosing( ), cTracker.getDurationTotalSecondsLoopClosing( )/dDurationTotal );
        std::printf( "[0](main) duration      Optimization: %7.2fs (%4.2f)\n", cTracker.getDurationTotalSecondsOptimization( ), cTracker.getDurationTotalSecondsOptimization( )/dDurationTotal );

        std::printf( "\n[0](main) distance traveled: %fm\n", dDistance );
        std::printf( "[0](main) traveling speed (avg): %fm/s\n", dDistance/dDurationRealTime );

        std::printf( "\n[0](main) total frames: %lu\n", uFrameCount );
        std::printf( "[0](main) invalid frames: %li (%4.2f)\n", uInvalidFrames, static_cast< double >( uInvalidFrames )/uFrameCount );
        CLogger::closeBox( );
    }
    else
    {
        std::printf( "[0](main) dataset completed - no frames processed\n" );
    }

    //ds exit
    return 0;
}

void setParametersNaive( const int& p_iArgc,
                         char** const p_pArgv,
                         std::string& p_strMode,
                         std::string& p_strInfileCameraIMUMessages )
{
    //ds attribute names (C style for printf)
    const char* arrParameter1 = "-mode";
    const char* arrParameter2 = "-messages";
    const char* arrParameter3 = "-h";
    const char* arrParameter4 = "--h";
    const char* arrParameter5 = "-help";
    const char* arrParameter6 = "--help";

    try
    {
        //ds parse optional command line arguments
        std::vector< std::string > vecCommandLineArguments;
        for( uint32_t u = 1; u < static_cast< uint32_t >( p_iArgc ); ++u )
        {
            //ds get parameter to string
            const std::string strParameter( p_pArgv[u] );

            //ds find '=' sign
            const std::string::size_type uStart( strParameter.find( '=' ) );

            vecCommandLineArguments.push_back( strParameter.substr( 0, uStart ) );
            vecCommandLineArguments.push_back( strParameter.substr( uStart+1, strParameter.length( )-uStart ) );
        }

        //ds check possible parameters in the vectorized command line arguments
        const std::vector< std::string >::const_iterator itParameter1( std::find( vecCommandLineArguments.begin( ), vecCommandLineArguments.end( ), arrParameter1 ) );
        const std::vector< std::string >::const_iterator itParameter2( std::find( vecCommandLineArguments.begin( ), vecCommandLineArguments.end( ), arrParameter2 ) );
        const std::vector< std::string >::const_iterator itParameter3( std::find( vecCommandLineArguments.begin( ), vecCommandLineArguments.end( ), arrParameter3 ) );
        const std::vector< std::string >::const_iterator itParameter4( std::find( vecCommandLineArguments.begin( ), vecCommandLineArguments.end( ), arrParameter4 ) );
        const std::vector< std::string >::const_iterator itParameter5( std::find( vecCommandLineArguments.begin( ), vecCommandLineArguments.end( ), arrParameter5 ) );
        const std::vector< std::string >::const_iterator itParameter6( std::find( vecCommandLineArguments.begin( ), vecCommandLineArguments.end( ), arrParameter6 ) );

        //ds check for help parameters first
        if( vecCommandLineArguments.end( ) != itParameter3 ||
            vecCommandLineArguments.end( ) != itParameter4 ||
            vecCommandLineArguments.end( ) != itParameter5 ||
            vecCommandLineArguments.end( ) != itParameter6 )
        {
            //ds print help and exit here
            printHelp( );
            std::exit( 0 );
        }

        //ds set parameters if found
        if( vecCommandLineArguments.end( ) != itParameter1 ){ p_strMode                    = *( itParameter1+1 ); }
        if( vecCommandLineArguments.end( ) != itParameter2 ){ p_strInfileCameraIMUMessages = *( itParameter2+1 ); }
    }
    catch( const std::invalid_argument& p_cException )
    {
        std::printf( "[0](setParametersNaive) malformed command line syntax\n" );
        printHelp( );
        std::exit( -1 );
    }
    catch( const std::out_of_range& p_cException )
    {
        std::printf( "[0](setParametersNaive) malformed command line syntax\n" );
        printHelp( );
        std::exit( -1 );
    }
}

void printHelp( )
{
    std::printf( "[0](printHelp) usage: tracker_sv -messages='textfile_path' [-mode='interactive'|'stepwise'|'benchmark']\n" );
}
