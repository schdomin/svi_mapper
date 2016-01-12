#include <opencv/highgui.h>
#include <stack>
#include <thread>

//ds custom
#include "txt_io/message_reader.h"
#include "core/CTrackerStereo.h"
#include "core/CMapper.h"
#include "exceptions/CExceptionLogfileTree.h"
#include "utility/CIMUInterpolator.h"
#include "utility/CParameterBase.h"
#include "optimization/Cg2oOptimizer.h"



//ds command line parsing (setting params IN/OUT)
void setParametersNaive( const int& p_iArgc,
                         char** const p_pArgv,
                         std::string& p_strMode,
                         std::string& p_strInfileCameraIMUMessages );

void printHelp( );

//ds speed tests
void testSpeedEigen( );



//ds C+11 threading - global scope
std::shared_ptr< CHandleLandmarks > g_hLandmarks( std::make_shared< CHandleLandmarks >( ) );
std::shared_ptr< CHandleKeyFrames > g_hKeyFrames( std::make_shared< CHandleKeyFrames >( ) );
std::shared_ptr< CHandleMapping > g_hMapper( std::make_shared< CHandleMapping >( ) );
std::shared_ptr< CHandleOptimization > g_hOptimizer( std::make_shared< CHandleOptimization >( ) );
std::shared_ptr< CHandleTracking > g_hTracker( std::make_shared< CHandleTracking >( ) );

//ds mapping thread
void launchMapping( )
{
    std::printf( "[1](launchMapping) thread launched\n" );

    //ds entities
    CMapper cMapper( g_hLandmarks, g_hKeyFrames, g_hMapper, g_hOptimizer );

    //ds acquire locks (for overview in this scope)
    std::unique_lock< std::mutex > cLockMapper( g_hMapper->cMutex, std::defer_lock );
    std::unique_lock< std::mutex > cLockLandmarks( g_hLandmarks->cMutex, std::defer_lock );
    std::unique_lock< std::mutex > cLockKeyFrames( g_hKeyFrames->cMutex, std::defer_lock );

    //ds breaking
    while( true )
    {
        //ds lock the mutex and wait for a call (new key frames, map update or termination)
        cLockMapper.lock( );
        g_hMapper->cConditionVariable.wait( cLockMapper, []{ return ( !g_hMapper->bWaitingForOptimizationReception && ( !g_hMapper->vecKeyFramesToAdd.empty( )||
                                                                                                                             g_hMapper->bTerminationRequested ||
                                                                                                                    g_hMapper->cMapUpdate.bAvailableForMapper ) ); } );

        //std::cerr << "mapper: in" <<  std::endl;

        //ds if termination requested
        if( g_hMapper->bTerminationRequested )
        {
            std::printf( "[1](launchMapping) termination request signal received\n" );
            g_hMapper->bActive = false;
            cLockMapper.unlock( );
            g_hMapper->cConditionVariable.notify_all( );
            break;
        }

        //ds check if a map update has been broadcasted from the optimizer
        if( g_hMapper->cMapUpdate.bAvailableForMapper )
        {
            //ds update map
            assert( !g_hMapper->cMapUpdate.bAvailableForTracker );
            cLockKeyFrames.lock( );
            cMapper.updateMap( );
            cLockKeyFrames.unlock( );

            //ds update processed
            g_hMapper->cMapUpdate.bAvailableForMapper   = false;
            g_hMapper->bWaitingForOptimizationReception = false;
        }

        //ds trigger optimization if required (fast)
        cLockLandmarks.lock( );
        cLockKeyFrames.lock( );
        const bool bOptimizationRequired = cMapper.checkAndRequestOptimization( );
        cLockKeyFrames.unlock( );
        cLockLandmarks.unlock( );

        //ds block further adding of key frames while optimizer has not received our request
        g_hMapper->bWaitingForOptimizationReception = bOptimizationRequired;

        //ds check if we can add new key frames
        if( !bOptimizationRequired )
        {
            //ds copy new key frames to buffer fast!
            cMapper.addKeyFramesSorted( g_hMapper->vecKeyFramesToAdd );

            //ds clear pending frames to add
            g_hMapper->vecKeyFramesToAdd.clear( );
        }

        //ds unlock mutex and notify main over the condition variable
        cLockMapper.unlock( );
        g_hMapper->cConditionVariable.notify_all( );

        //ds process new key frames after critical section (internal race condition with optimization thread) - if not waiting for an optimization
        if( !bOptimizationRequired )
        {
            //ds lock - blocking - RAII
            cLockKeyFrames.lock( );
            cMapper.integrateAddedKeyFrames( );
            cLockKeyFrames.unlock( );
        }

        //std::cerr << "mapper: out" << std::endl;
    }

    //ds report
    std::printf( "[1](launchMapping) thread terminated\n" );
}

//ds optimization thread
void launchOptimization( const std::shared_ptr< CStereoCamera > p_pCameraSTEREO, const Eigen::Isometry3d p_matTransformationWORLDtoLEFTInitial )
{
    std::printf( "[2](launchOptimization) thread launched\n" );

    //ds allocate optimizer
    Cg2oOptimizer cOptimizer( p_pCameraSTEREO,
                              g_hLandmarks,
                              g_hKeyFrames,
                              g_hMapper,
                              g_hTracker,
                              p_matTransformationWORLDtoLEFTInitial );

    //ds acquire locks (for overview in this scope)
    std::unique_lock< std::mutex > cLockOptimizer( g_hOptimizer->cMutex, std::defer_lock );

    //ds breaking
    while( true )
    {
        //ds lock the mutex and wait for a call (new optimization requests or termination)
        cLockOptimizer.lock( );
        g_hOptimizer->cConditionVariable.wait( cLockOptimizer, []{ return ( g_hOptimizer->bTerminationRequested || !g_hOptimizer->bRequestProcessed ); } );

        //std::cerr << "optimizer: in" <<  std::endl;

        //ds if termination requested
        if( g_hOptimizer->bTerminationRequested )
        {
            std::printf( "[2](launchOptimization) termination request signal received\n" );
            g_hOptimizer->bActive = false;
            cLockOptimizer.unlock( );
            g_hOptimizer->cConditionVariable.notify_one( );
            break;
        }

        //ds run optimization based on request (heavy INNER LOCKING)
        cOptimizer.optimize( g_hOptimizer->cRequest );
        g_hOptimizer->bRequestProcessed = true;

        //ds unlock mutex and notify main over the condition variable
        cLockOptimizer.unlock( );
        g_hOptimizer->cConditionVariable.notify_one( );

        //std::cerr << "optimizer: out" <<  std::endl;
    }

    std::printf( "[2](launchOptimization) thread terminated\n" );
}

int32_t main( int32_t argc, char **argv )
{
    //assert( false );

    //ds thread control
    std::unique_lock< std::mutex > cLockMapper( g_hMapper->cMutex, std::defer_lock );
    std::unique_lock< std::mutex > cLockOptimizer( g_hOptimizer->cMutex, std::defer_lock );
    std::unique_lock< std::mutex > cLockTracker( g_hTracker->cMutex );
    g_hTracker->bBusy = true;

    //ds defaults
    std::string strMode              = "benchmark";
    std::string strInfileMessageDump = "";
    std::string strConfigurationCameraLEFT  = "../hardware_parameters/vi_sensor_camera_left.txt";
    std::string strConfigurationCameraRIGHT = "../hardware_parameters/vi_sensor_camera_right.txt";

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
        CParameterBase::constructCameraSTEREO( );
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

    //ds evaluate IMU situation first
    std::shared_ptr< CIMUInterpolator > pIMUInterpolator( std::make_shared< CIMUInterpolator >( ) );

    //ds stop time
    const double dTimeStartSeconds = CLogger::getTimeSeconds( );

    //ds message holders
    std::shared_ptr< txt_io::CIMUMessage > pMessageIMU( 0 );
    std::shared_ptr< txt_io::PinholeImageMessage > pMessageCameraLEFT( 0 );
    std::shared_ptr< txt_io::PinholeImageMessage > pMessageCameraRIGHT( 0 );

    //ds playback the dump - IMU calibration
    while( cMessageReader.good( ) && !pIMUInterpolator->isCalibrated( ) )
    {
        //ds retrieve a message
        txt_io::BaseMessage* msgBase = cMessageReader.readMessage( );

        //ds if set
        if( 0 != msgBase )
        {
            //ds trigger callbacks artificially - check for imu input first
            if( "IMU_MESSAGE" == msgBase->tag( ) )
            {
                //ds IMU message
                pMessageIMU = std::shared_ptr< txt_io::CIMUMessage >( dynamic_cast< txt_io::CIMUMessage* >( msgBase ) );

                //ds add to interpolator
                pIMUInterpolator->addMeasurementCalibration( pMessageIMU->getLinearAcceleration( ), pMessageIMU->getAngularVelocity( ) );
            }
        }
    }

    //ds must be calibrated
    assert( pIMUInterpolator->isCalibrated( ) );
    assert( 0 != CParameterBase::pCameraLEFT );
    assert( 0 != CParameterBase::pCameraRIGHT );
    assert( 0 != CParameterBase::pCameraSTEREO );

    //ds allocate the tracker
    CTrackerStereo cTracker( CParameterBase::pCameraSTEREO,
                             pIMUInterpolator,
                             g_hLandmarks,
                             g_hMapper,
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

    //ds spawn worker threads
    std::thread threadMapping( launchMapping );
    std::thread threadOptimization( launchOptimization, CParameterBase::pCameraSTEREO, pIMUInterpolator->getTransformationWORLDtoCAMERA( CParameterBase::pCameraLEFT->m_matRotationIMUtoCAMERA ) );
    g_hTracker->bBusy = false;
    cLockTracker.unlock( );
    g_hTracker->cConditionVariable.notify_one( );

    //ds playback the dump
    while( cMessageReader.good( ) && !cTracker.isShutdownRequested( ) )
    {
        //ds retrieve a message
        txt_io::BaseMessage* msgBase = cMessageReader.readMessage( );

        //ds if set
        if( 0 != msgBase )
        {
            //ds trigger callbacks artificially - check for imu input first
            if( "IMU_MESSAGE" == msgBase->tag( ) )
            {
                //ds IMU message
                pMessageIMU = std::shared_ptr< txt_io::CIMUMessage >( dynamic_cast< txt_io::CIMUMessage* >( msgBase ) );

                //ds add to interpolator
                pIMUInterpolator->addMeasurement( pMessageIMU->getLinearAcceleration( ), pMessageIMU->getAngularVelocity( ) );
            }
            else
            {
                //ds camera message
                std::shared_ptr< txt_io::PinholeImageMessage > pMessageImage( dynamic_cast< txt_io::PinholeImageMessage* >( msgBase ) );

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
        }

        //ds as soon as we have data in all the stacks - process
        if( 0 != pMessageCameraLEFT && 0 != pMessageCameraRIGHT && 0 != pMessageIMU )
        {
            //ds if the timestamps match (optimally the case)
            if( pMessageCameraLEFT->timestamp( ) == pMessageCameraRIGHT->timestamp( ) )
            {
                //ds synchronization expected
                assert( pMessageIMU->timestamp( ) == pMessageCameraLEFT->timestamp( ) );
                assert( pMessageIMU->timestamp( ) == pMessageCameraRIGHT->timestamp( ) );

                //ds callback with triplet (locking busy tracker so optimization application will wait) - or getting blocked while optimization is applying update
                cLockTracker.lock( );
                g_hTracker->bBusy = true;

                //std::cerr << "tracker: in" << std::endl;

                //ds check if there's a map update available
                cLockMapper.lock( );
                if( g_hMapper->cMapUpdate.bAvailableForTracker )
                {
                    //ds signal optimizer that we received the update
                    assert( !g_hMapper->cMapUpdate.bAvailableForMapper );
                    g_hMapper->cMapUpdate.bAvailableForTracker = false;
                    const CMapUpdate cUpdate( g_hMapper->cMapUpdate );
                    cLockMapper.unlock( );
                    g_hMapper->cConditionVariable.notify_all( );

                    //std::cerr << "tracker: received update" << std::endl;

                    {
                        //ds process update
                        std::lock_guard< std::mutex > cLockLandmarks( g_hLandmarks->cMutex );
                        std::lock_guard< std::mutex > cLockKeyFrames( g_hKeyFrames->cMutex );
                        cTracker.updateMap( cUpdate );
                    }

                    //ds lock again to enable mapper update
                    cLockMapper.lock( );
                    g_hMapper->cMapUpdate.bAvailableForMapper = true;
                    cLockMapper.unlock( );
                    g_hMapper->cConditionVariable.notify_all( );
                }
                else
                {
                    cLockMapper.unlock( );
                }

                //ds evaluate images and IMU (inner landmark locking)
                cTracker.receivevDataVI( pMessageCameraLEFT, pMessageCameraRIGHT, pMessageIMU );
                g_hTracker->bBusy = false;
                cLockTracker.unlock( );
                g_hTracker->cConditionVariable.notify_one( );

                //std::cerr << "tracker: out" << std::endl;

                //ds reset holders
                pMessageCameraLEFT.reset( );
                pMessageCameraRIGHT.reset( );
                pMessageIMU.reset( );

                //ds check reset
                assert( 0 == pMessageCameraLEFT );
                assert( 0 == pMessageCameraRIGHT );
                assert( 0 == pMessageIMU );
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

    //ds signal threads: mapper
    CLogger::openBox( );
    std::printf( "[0](main) signaling threads for termination\n" );
    cLockMapper.lock( );
    g_hMapper->bTerminationRequested = true;
    cLockMapper.unlock( );
    g_hMapper->cConditionVariable.notify_one( );
    cLockMapper.lock( );
    g_hMapper->cConditionVariable.wait( cLockMapper, []{ return !g_hMapper->bActive; } );
    cLockMapper.unlock( );

    //ds optimizer
    cLockOptimizer.lock( );
    g_hOptimizer->bTerminationRequested = true;
    cLockOptimizer.unlock( );
    g_hOptimizer->cConditionVariable.notify_one( );
    cLockOptimizer.lock( );
    g_hOptimizer->cConditionVariable.wait( cLockOptimizer, []{ return !g_hOptimizer->bActive; } );
    cLockOptimizer.unlock( );

    //ds join threads
    threadMapping.join( );
    threadOptimization.join( );

    std::printf( "[0](main) all threads shut down successfully\n" );
    CLogger::closeBox( );

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

    //ds speed checks
    //testSpeedEigen( );

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
    std::printf( "[0](printHelp) usage: stereo_fps -messages='textfile_path' [-mode='interactive'|'stepwise'|'benchmark']\n" );
}

void testSpeedEigen( )
{
    //ds build speed check
    CLogger::openBox( );
    std::printf( "[0](testSpeedEigen) speed test started - this might take a while\n" );
    Eigen::Matrix< double, 100, 100 > matTest( Eigen::Matrix< double, 100, 100 >::Identity( ) );
    const double dTimeStartSeconds = CTimer::getTimeSeconds( );
    for( uint64_t u = 0; u < 1e5; ++u )
    {
        matTest = matTest*matTest;
        for( uint8_t v = 0; v < matTest.rows( ); ++v )
        {
            for( uint8_t w = 0; w < matTest.cols( ); ++w )
            {
                assert( 0 <= matTest(v,w) );
            }
        }
    }
    std::printf( "[0](testSpeedEigen) speed test run complete - duration: %fs\n", CTimer::getTimeSeconds( )-dTimeStartSeconds );
    CLogger::closeBox( );
}
