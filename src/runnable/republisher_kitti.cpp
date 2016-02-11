#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv/highgui.h>
#include <sys/stat.h>

//ds ROS
//#include <ros/ros.h>
//#include <image_transport/image_transport.h>
//#include <cv_bridge/cv_bridge.h>

//ds custom
#include "txt_io/pinhole_image_message.h"
#include "utility/CLogger.h"

//ds fake session counters
uint64_t g_uFrameIDCameraRIGHT = 0;
uint64_t g_uFrameIDCameraLEFT  = 0;

//ds message buffer
std::ofstream g_strOutfile;

inline void readNextMessageFromFile( const double& p_dTimestampSeconds,
                                     const std::string& p_strImageFolderLEFT,
                                     const std::string& p_strImageFolderRIGHT,
                                     const std::string& p_strOutfile );

int32_t main( int32_t argc, char **argv )
{
    //ds pwd info
    std::printf( "(main) launched: %s\n", argv[0] );

    if( 3 != argc )
    {
        std::printf( "(main) <KITTI sequence path> <outfile path>\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    //ds default files
    std::string strInfileTimestamps = argv[1]; strInfileTimestamps += "/times.txt";
    std::string strImageFolderLEFT  = argv[1]; strImageFolderLEFT += "/image_0/";
    std::string strImageFolderRIGHT = argv[1]; strImageFolderRIGHT += "/image_1/";
    std::string strOutfile          = argv[2];

    //ds open outfile
    g_strOutfile.open( strOutfile, std::ofstream::out );

    //ds on failure
    if( !g_strOutfile.good( ) )
    {
        std::printf( "(main) unable to create outfile\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    //ds create image file directory
    const std::string strOutfileDirectory( strOutfile+".d/" );
    if( 0 != mkdir( strOutfileDirectory.c_str( ), 0700 ) )
    {
        std::printf( "(main) unable to create image directory\n" );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    //ds open the timestamp file
    std::ifstream ifTimestamps( strInfileTimestamps, std::ifstream::in );

    //ds timestamps (changing)
    std::vector< double > vecTimestampsSeconds;

    try
    {
        //ds compute timing
        while( ifTimestamps.good( ) )
        {
            //ds line buffer
            std::string strLineBuffer;

            //ds read one line
            std::getline( ifTimestamps, strLineBuffer );

            //ds check if nothing was read
            if( strLineBuffer.empty( ) )
            {
                //ds escape
                break;
            }
            else
            {
                //ds add it to the vector
                vecTimestampsSeconds.push_back( std::stod( strLineBuffer ) );
            }
        }
    }
    catch( const std::exception& p_cException )
    {
        //ds halt on any exception
        std::printf( "\n(main) ERROR: unable to parse timestamps file, exception: '%s'\n", p_cException.what( ) );
        std::printf( "(main) terminated: %s\n", argv[0] );
        std::fflush( stdout );
        return 1;
    }

    assert( 1 < vecTimestampsSeconds.size( ) );
    std::printf( "(main) successfully loaded timestamps: %lu\n", vecTimestampsSeconds.size( ) );

    //ds log configuration
    CLogger::openBox( );
    //std::printf( "(main) ROS Node namespace   := '%s'\n", hNode.getNamespace( ).c_str( ) );
    std::printf( "(main) strInfileTimestamps  := '%s'\n", strInfileTimestamps.c_str( ) );
    std::printf( "(main) strImageFolderLEFT   := '%s'\n", strImageFolderLEFT.c_str( ) );
    std::printf( "(main) strImageFolderRIGHT  := '%s'\n", strImageFolderRIGHT.c_str( ) );
    std::printf( "(main) strOutfile           := '%s'\n", strOutfile.c_str( ) );
    std::fflush( stdout );
    CLogger::closeBox( );

    std::printf( "(main) press [ENTER] to start playback\n" );
    while( -1 == getchar( ) )
    {
        usleep( 1 );
    }
    std::printf( "\n(main) streaming to file\n" );

    //ds playback the dump
    for( uint64_t u = 0; u < vecTimestampsSeconds.size( ); ++u )
    {
        //ds read a message
        readNextMessageFromFile( vecTimestampsSeconds[u], strImageFolderLEFT, strImageFolderRIGHT, strOutfileDirectory );

        //ds info
        std::printf( "remaining time: %f\n", vecTimestampsSeconds.back( )-vecTimestampsSeconds[u] );
        std::fflush( stdout );
    }

    //ds done
    g_strOutfile.close( );

    //ds exit
    std::printf( "(main) terminated: %s\n", argv[0] );
    std::fflush( stdout);
    return 0;
}

inline void readNextMessageFromFile( const double& p_dTimestampSeconds,
                                     const std::string& p_strImageFolderLEFT,
                                     const std::string& p_strImageFolderRIGHT,
                                     const std::string& p_strOutfile )
{
    //ds parse LEFT image - build image file name
    char chBufferLEFT[10];
    std::snprintf( chBufferLEFT, 7, "%06lu", g_uFrameIDCameraLEFT );
    const std::string strImageFileLEFT = p_strImageFolderLEFT + chBufferLEFT + ".png";

    //ds read the image
    cv::Mat matImageLEFT = cv::imread( strImageFileLEFT, cv::IMREAD_GRAYSCALE );

    //ds parse RIGHT image - build image file name
    char chBufferRIGHT[10];
    std::snprintf( chBufferRIGHT, 7, "%06lu", g_uFrameIDCameraRIGHT );
    const std::string strImageFileRIGHT = p_strImageFolderRIGHT + chBufferRIGHT + ".png";

    //ds read the image
    cv::Mat matImageRIGHT = cv::imread( strImageFileRIGHT, cv::IMREAD_GRAYSCALE );

    //ds synchronization enforced
    assert( g_uFrameIDCameraLEFT == g_uFrameIDCameraRIGHT );

    //ds create pinhole messages
    txt_io::PinholeImageMessage cMessageLEFT( "/thin_visensor_node/camera_left/image_raw", "camera_left", g_uFrameIDCameraLEFT, p_dTimestampSeconds );
    txt_io::PinholeImageMessage cMessageRIGHT( "/thin_visensor_node/camera_right/image_raw", "camera_right", g_uFrameIDCameraRIGHT, p_dTimestampSeconds );

    //ds set images
    cMessageLEFT.setBinaryFilePrefix( p_strOutfile );
    cMessageRIGHT.setBinaryFilePrefix( p_strOutfile );
    cMessageLEFT.setImage( matImageLEFT );
    cMessageRIGHT.setImage( matImageRIGHT );

    //ds write to stream
    g_strOutfile << "PINHOLE_IMAGE_MESSAGE ";
    cMessageLEFT.toStream( g_strOutfile );
    g_strOutfile << "\nPINHOLE_IMAGE_MESSAGE ";
    cMessageRIGHT.toStream( g_strOutfile );
    g_strOutfile << "\n";

    //ds publish the images
    ++g_uFrameIDCameraLEFT;
    ++g_uFrameIDCameraRIGHT;
}
