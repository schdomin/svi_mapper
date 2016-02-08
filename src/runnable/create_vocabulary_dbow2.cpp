#include <stdio.h>

//ds custom
#include "txt_io/pinhole_image_message.h"
#include "txt_io/message_reader.h"
#include "utility/CParameterBase.h"
#include "DBoW2.h" // defines Surf64Vocabulary and Surf64Database
#include "DUtils/DUtils.h"
#include "DUtilsCV/DUtilsCV.h" // defines macros CVXX
#include "DVision/DVision.h"

#define DESCRIPTOR_SIZE_BITS 256


int32_t main( int32_t argc, char** argv )
{
    std::printf( "(main) launched: '%s'\n", argv[0] );

    //ds check params
    if( 2 > argc )
    {
        std::printf( "(main) invalid call, please provide at least: <message_file>\n" );
        return -1;
    }

    //ds feature handling
    std::shared_ptr< cv::GoodFeaturesToTrackDetector > pDetector = std::make_shared< cv::GoodFeaturesToTrackDetector >( 1000, 0.01, 7.0, 7, true );
    std::shared_ptr< cv::BriefDescriptorExtractor > pExtractor   = std::make_shared< cv::BriefDescriptorExtractor >( DESCRIPTOR_SIZE_BITS/8 );

    //ds vocabulary settings
    const uint32_t uBranchingFactor = 10;
    const uint32_t uDepthLevels = 6;
    uint64_t uNumberOfDescriptorsTotal = 0;

    //ds instantiate controller
    BriefVocabulary cVoc( uBranchingFactor, uDepthLevels, DBoW2::BINARY );

    //ds data structure containing key frames with descriptor info
    std::vector< std::vector< boost::dynamic_bitset< > > > vecDescriptorsPerFrames;

    //ds for each input message folder
    for( uint32_t uMessagesFile = 1; uMessagesFile < static_cast< uint32_t >( argc ); ++uMessagesFile )
    {
        //ds set file name
        const std::string strMessageFile = argv[uMessagesFile];
        std::printf( "(main) strMessageFile := '%s'\n", strMessageFile.c_str( ) );

        //ds message loop
        txt_io::MessageReader cMessageReader;
        cMessageReader.open( strMessageFile );

        //ds escape here on failure
        if( !cMessageReader.good( ) )
        {
            std::printf( "(main) unable to open message file: '%s'\n", strMessageFile.c_str( ) );
            return -1;
        }

        try
        {
            //ds load camera parameters
            CParameterBase::loadCameraLEFTwithIMU( "../hardware_parameters/vi_sensor_camera_left.txt" );
            std::printf( "(main) successfully imported camera LEFT\n" );
            CParameterBase::loadCameraRIGHTwithIMU( "../hardware_parameters/vi_sensor_camera_right.txt" );
            std::printf( "(main) successfully imported camera RIGHT\n" );
            CParameterBase::constructCameraSTEREOwithIMU( );
        }
        catch( const CExceptionParameter& p_cException )
        {
            std::printf( "(main) unable to import camera parameters - CExceptionParameter: '%s'\n", p_cException.what( ) );
            return 1;
        }
        catch( const std::invalid_argument& p_cException )
        {
            std::printf( "(main) unable to import camera parameters - std::invalid_argument: '%s'\n", p_cException.what( ) );
            return 1;
        }
        catch( const std::out_of_range& p_cException )
        {
            std::printf( "(main) unable to import camera parameters - std::out_of_range: '%s'\n", p_cException.what( ) );
            return 1;
        }

        //ds message holders
        std::shared_ptr< txt_io::PinholeImageMessage > pMessageCameraLEFT( 0 );
        std::shared_ptr< txt_io::PinholeImageMessage > pMessageCameraRIGHT( 0 );

        //ds playback the dump
        while( cMessageReader.good( ) )
        {
            //ds retrieve a message
            txt_io::BaseMessage* msgBase = cMessageReader.readMessage( );

            //ds if set
            if( 0 != msgBase )
            {
                //ds trigger callbacks artificially - check for imu input first
                if( "PINHOLE_IMAGE_MESSAGE" == msgBase->tag( ) )
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

                    //ds as soon as we have data in all the stacks - process
                    if( 0 != pMessageCameraLEFT && 0 != pMessageCameraRIGHT )
                    {
                        //ds if the timestamps match (optimally the case)
                        if( pMessageCameraLEFT->timestamp( ) == pMessageCameraRIGHT->timestamp( ) )
                        {
                            //ds preprocessed image holders
                            cv::Mat matPreprocessedLEFT( pMessageCameraLEFT->image( ) );
                            cv::Mat matPreprocessedRIGHT( pMessageCameraRIGHT->image( ) );

                            //ds preprocess images
                            cv::equalizeHist( pMessageCameraLEFT->image( ), matPreprocessedLEFT );
                            cv::equalizeHist( pMessageCameraRIGHT->image( ), matPreprocessedRIGHT );
                            CParameterBase::pCameraSTEREOwithIMU->undistortAndrectify( matPreprocessedLEFT, matPreprocessedRIGHT );

                            //ds key points buffer
                            std::vector< cv::KeyPoint > vecKeyPoints;

                            //ds detect features
                            pDetector->detect( matPreprocessedLEFT, vecKeyPoints );

                            //ds compute descriptors for the keypoints
                            CDescriptors matDescriptors;
                            pExtractor->compute( matPreprocessedLEFT, vecKeyPoints, matDescriptors );

                            //ds current descriptor vector
                            std::vector< boost::dynamic_bitset< > > vecDescriptorPool;
                            vecDescriptorPool.reserve( vecKeyPoints.size( ) );

                            //ds fill descriptor vector
                            for( uint64_t u = 0; u < vecKeyPoints.size( ); ++u )
                            {
                                //ds buffer descriptor
                                const CDescriptor matDescriptorLEFT( matDescriptors.row( u ) );

                                //ds boost bitset
                                boost::dynamic_bitset< > vecDescriptor( DESCRIPTOR_SIZE_BITS );

                                //ds loop over all bytes
                                for( uint32_t v = 0; v < DESCRIPTOR_SIZE_BITS/8; ++v )
                                {
                                    //ds get minimal datafrom cv::mat
                                    const uchar chValue = matDescriptorLEFT.at< uchar >( v );

                                    //ds get bitstring
                                    for( uint8_t w = 0; w < 8; ++w )
                                    {
                                        vecDescriptor[v*8+w] = ( chValue >> w ) & 1;
                                    }
                                }

                                //ds add to descriptors
                                vecDescriptorPool.push_back( vecDescriptor );
                            }

                            //ds add to complete holder structure
                            vecDescriptorsPerFrames.push_back( vecDescriptorPool );
                            uNumberOfDescriptorsTotal += vecDescriptorPool.size( );

                            std::printf( "(main) [%s] frame: %06lu descriptors: %6lu/%lu\n", strMessageFile.c_str( ), vecDescriptorsPerFrames.size( ), vecDescriptorPool.size( ), uNumberOfDescriptorsTotal );
                        }

                        //ds reset holders
                        pMessageCameraLEFT.reset( );
                        pMessageCameraRIGHT.reset( );
                    }
                }
            }
        }
    }

    //ds create the vocabulary
    std::printf( "(main) creating vocabulary\n" );
    const double dTimeStartSeconds = CTimer::getTimeSeconds( );
    cVoc.create( vecDescriptorsPerFrames );
    const double dDurationSeconds = CTimer::getTimeSeconds( )-dTimeStartSeconds;
    std::printf( "(main) creation complete - duration: %fs\n", dDurationSeconds );

    //ds construct filename
    const std::string strVocabularyName( "vocabulary_BRIEF_"+std::to_string( uNumberOfDescriptorsTotal )
                                        +"_"+std::to_string( DESCRIPTOR_SIZE_BITS )
                                        +"_K"+std::to_string( uBranchingFactor )
                                        +"_L"+std::to_string( uDepthLevels )+".yml.gz" );

    //ds save to disk
    cVoc.save( strVocabularyName );
    std::printf( "(main) saved DBoW2 vocabulary to: '%s'\n", strVocabularyName.c_str( ) );

    //ds done
    std::printf( "(main) terminated: '%s'\n", argv[0] );
    return 0;
}
