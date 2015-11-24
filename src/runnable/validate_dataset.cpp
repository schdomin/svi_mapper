#include <assert.h>
#include <memory>



//ds custom
#include "utility/CTimer.h"
#include "txt_io/pinhole_image_message.h"
#include "txt_io/imu_message.h"
#include "txt_io/message_reader.h"



//ds command line parsing (setting params IN/OUT)
void setParametersNaive( const int& p_iArgc,
                         char** const p_pArgv,
                         std::string& p_strInfileCameraIMUMessages );

int32_t main( int32_t argc, char **argv )
{
    //ds pwd info
    std::printf( "(main) launched: %s\n", argv[0] );

    //ds defaults
    std::string strInfileMessageDump = "";

    //ds get params
    setParametersNaive( argc, argv, strInfileMessageDump );

    //ds escape here on failure
    if( strInfileMessageDump.empty( ) )
    {
        std::printf( "(main) no message file specified\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    //ds message loop
    txt_io::MessageReader cMessageReader;
    cMessageReader.open( strInfileMessageDump );

    //ds escape here on failure
    if( !cMessageReader.good( ) )
    {
        std::printf( "(main) unable to open message file: %s\n", strInfileMessageDump.c_str( ) );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    //ds log configuration
    std::printf( "(main) strInfileMessageDump := '%s'\n", strInfileMessageDump.c_str( ) );
    std::printf( "(main) starting test run\n" );
    std::fflush( stdout );

    //ds message holders
    std::shared_ptr< txt_io::CIMUMessage > pMessageIMU( 0 );
    std::shared_ptr< txt_io::PinholeImageMessage > pMessageCameraLEFT( 0 );
    std::shared_ptr< txt_io::PinholeImageMessage > pMessageCameraRIGHT( 0 );

    //ds counters
    uint64_t uCountMessagesIMU = 0;
    uint64_t uCountMessagesCameraLEFT  = 0;
    uint64_t uCountMessagesCameraRIGHT = 0;
    uint64_t uCountTriplets = 0;
    uint64_t uCountFailures = 0;


    //ds timing
    const double dTimeStartSeconds = CTimer::getTimeSeconds( );

    //ds playback the dump
    while( cMessageReader.good( ) )
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
                ++uCountMessagesIMU;
            }
            else
            {
                //ds camera message
                std::shared_ptr< txt_io::PinholeImageMessage > pMessageImage( dynamic_cast< txt_io::PinholeImageMessage* >( msgBase ) );

                //ds if its the left camera
                if( "camera_left" == pMessageImage->frameId( ) )
                {
                    pMessageCameraLEFT  = pMessageImage;
                    ++uCountMessagesCameraLEFT;
                }
                else
                {
                    pMessageCameraRIGHT = pMessageImage;
                    ++uCountMessagesCameraRIGHT;
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

                //ds reset holders
                pMessageCameraLEFT.reset( );
                pMessageCameraRIGHT.reset( );
                pMessageIMU.reset( );

                //ds check reset
                assert( 0 == pMessageCameraLEFT );
                assert( 0 == pMessageCameraRIGHT );
                assert( 0 == pMessageIMU );

                //ds worked
                ++uCountTriplets;
            }
            else
            {
                //ds check timestamp mismatch
                if( pMessageCameraLEFT->timestamp( ) < pMessageCameraRIGHT->timestamp( ) )
                {
                    std::printf( "(main) timestamp mismatch LEFT: %f < RIGHT: %f - processing skipped\n", pMessageCameraLEFT->timestamp( ), pMessageCameraRIGHT->timestamp( ) );
                    pMessageCameraLEFT.reset( );
                }
                else
                {
                    std::printf( "(main) timestamp mismatch LEFT: %f > RIGHT: %f - processing skipped\n", pMessageCameraLEFT->timestamp( ), pMessageCameraRIGHT->timestamp( ) );
                    pMessageCameraRIGHT.reset( );
                }

                //ds failure
                ++uCountFailures;
            }
        }
    }

    //ds timing
    const double dDurationPlaybackSeconds = CTimer::getTimeSeconds( )-dTimeStartSeconds;
    const double dFramesPerSecond         = uCountTriplets/dDurationPlaybackSeconds;

    //ds summary
    std::printf( "(main) dataset completed\n\n" );
    std::printf( "(main) duration: %f\n", dDurationPlaybackSeconds );
    std::printf( "(main)      fps: %f\n\n", dFramesPerSecond );
    std::printf( "(main) messages          IMU: %lu\n", uCountMessagesIMU );
    std::printf( "(main) messages camera  LEFT: %lu\n", uCountMessagesCameraLEFT );
    std::printf( "(main) messages camera RIGHT: %lu\n\n", uCountMessagesCameraRIGHT );
    std::printf( "(main)    good frames: %lu\n", uCountTriplets  );
    std::printf( "(main) invalid frames: %lu\n\n", uCountFailures );
    std::printf( "(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}

void setParametersNaive( const int& p_iArgc,
                         char** const p_pArgv,
                         std::string& p_strInfileCameraIMUMessages )
{
    //ds attribute names (C style for printf)
    const char* arrParameter2 = "-messages";

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

        //ds check possible parameters
        const std::vector< std::string >::const_iterator itParameter2( std::find( vecCommandLineArguments.begin( ), vecCommandLineArguments.end( ), arrParameter2 ) );

        //ds set parameters if found
        if( vecCommandLineArguments.end( ) != itParameter2 ){ p_strInfileCameraIMUMessages = *( itParameter2+1 ); }
    }
    catch( const std::invalid_argument& p_cException )
    {
        std::printf( "(setParametersNaive) malformed command line syntax, usage: validate_dataset %s\n", arrParameter2 );
        std::printf( "(setParametersNaive) terminated: %s\n", p_pArgv[0] );
        std::fflush( stdout );
        exit( -1 );
    }
    catch( const std::out_of_range& p_cException )
    {
        std::printf( "(setParametersNaive) malformed command line syntax, usage: validate_dataset %s\n", arrParameter2 );
        std::printf( "(setParametersNaive) terminated: %s\n", p_pArgv[0] );
        std::fflush( stdout );
        exit( -1 );
    }
}
