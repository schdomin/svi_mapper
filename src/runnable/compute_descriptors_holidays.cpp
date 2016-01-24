#include <opencv/highgui.h>
#include <dirent.h>



#include "../types/Typedefs.h"
#include "../utility/CWrapperOpenCV.h"
#include "../utility/CLogger.h"



int32_t main( int32_t argc, char** argv )
{
    //ds catch invalid parameter count
    if( 5 != argc )
    {
        std::printf( "(main) invalid call - parameters: <folder_images> <file_baseline> <clouds_out> <query_clouds_out>\n" );
        std::fflush( stdout);
        return -1;
    }

    //ds folders
    const std::string strFolderImages( argv[1] );
    const std::string strFileBaseline( argv[2] );

    //ds load baseline file
    std::ifstream ifBaseline( strFileBaseline, std::ifstream::in );
    std::set< std::string > vecQueries;

    //ds read the file
    while( ifBaseline.good( ) )
    {
        //ds parse query image
        std::string strLine( "" );
        std::getline( ifBaseline, strLine );
        if( 0 < strLine.substr( 0, 10 ).length( ) )
        {
            vecQueries.insert( strLine.substr( 0, 10 ) );
        }
    }

    std::printf( "(main) successfully loaded queries: %lu\n", vecQueries.size( ) );






    //ds final clouds
    std::vector< std::pair< uint64_t, CCloudDescriptorBRIEF > > vecCloudsTotal( 0 );
    std::vector< std::pair< uint64_t, CCloudDescriptorBRIEF > > vecCloudsQuery( 0 );

    //ds descriptor detection
    cv::GoodFeaturesToTrackDetector cDetector( 10000, 0.01, 7.0, 7, true );

    //ds descriptor extraction
    cv::BriefDescriptorExtractor cExtractor( 32 );

    //ds total descriptor counts
    uint64_t uNumberOfDescriptors = 0;
    uint64_t uNumberOfDescriptorsQuery = 0;

    DIR *cDirectory = 0;
    dirent *cFile = 0;
    if( 0 != ( cDirectory = opendir( strFolderImages.c_str( ) ) ) )
    {
        while( 0 != ( cFile = readdir( cDirectory ) ) )
        {
            //ds full filepath
            const std::string strFile( strFolderImages+cFile->d_name );

            //ds try to load image
            cv::Mat matImage( cv::imread( strFile, CV_LOAD_IMAGE_GRAYSCALE ) );   // Read the file

            //ds if we have data
            if( 0 != matImage.data )
            {
                //ds current descriptor cloud
                CCloudDescriptorBRIEF cCloud( 0 );

                //ds key points buffer
                std::vector< cv::KeyPoint > vecKeyPoints;

                //ds detect features -> keypoints
                cDetector.detect( matImage, vecKeyPoints );

                //ds compute descriptors for the keypoints
                CDescriptors matDescriptors;
                cExtractor.compute( matImage, vecKeyPoints, matDescriptors );
                uNumberOfDescriptors += vecKeyPoints.size( );

                //ds parse descriptors to cloud
                for( std::vector< cv::KeyPoint >::size_type u = 0; u < vecKeyPoints.size( ); ++u )
                {
                    //ds add to cloud in our format
                    cCloud.push_back( CWrapperOpenCV::getDescriptorBRIEF( matDescriptors.row( u ) ) );
                }

                //ds parse query filename to id
                const std::string strFilename( cFile->d_name );
                const uint64_t uID = std::stoi( strFilename.substr( 0, 6 ) );

                //ds check if query
                if( 0 < vecQueries.count( strFilename ) )
                {
                    //ds add cloud to queries as well
                    vecCloudsQuery.push_back( std::make_pair( uID, cCloud ) );
                    uNumberOfDescriptorsQuery += vecKeyPoints.size( );
                }

                //ds add cloud
                vecCloudsTotal.push_back( std::make_pair( uID, cCloud ) );

                //ds info
                if( 0 < vecQueries.count( strFilename ) )
                {
                    std::printf( "(main) processed image: '%s' (%03lu|%06lu) QUERY\n", strFile.c_str( ), cCloud.size( ), vecCloudsTotal.size( ) );
                }
                else
                {
                    std::printf( "(main) processed image: '%s' (%03lu|%06lu)\n", strFile.c_str( ), cCloud.size( ), vecCloudsTotal.size( ) );
                }
            }
            else
            {
                std::printf( "(main) unable to load image: '%s'\n", strFile.c_str( ) );
            }
        }
        closedir( cDirectory );
    }
    else
    {
        std::printf( "(main) unable to open image folder: '%s'\n", argv[1] );
        std::fflush( stdout);
        return 1;
    }

    std::printf( "(main) parsed total clouds: %lu (total descriptors: %lu)\n", vecCloudsTotal.size( ), uNumberOfDescriptors );
    std::printf( "(main) parsed query clouds: %lu (total descriptors: %lu)\n", vecCloudsQuery.size( ), uNumberOfDescriptorsQuery );

    //ds construct file names
    std::ofstream ofCloud( argv[3], std::ofstream::out );
    std::ofstream ofCloudQuery( argv[4], std::ofstream::out );

    std::printf( "(main) saving total clouds to: '%s'\n", argv[3] );
    std::printf( "(main) saving query clouds to: '%s'\n", argv[4] );

    //ds store clouds number
    CLogger::writeDatum( ofCloud, vecCloudsTotal.size( ) );

    //ds save total clouds to file
    for( const std::pair< uint64_t, CCloudDescriptorBRIEF> cCloud: vecCloudsTotal )
    {
        //ds write cloud id
        CLogger::writeDatum( ofCloud, cCloud.first );

        //ds store number of descriptors
        CLogger::writeDatum( ofCloud, cCloud.second.size( ) );

        //ds each descriptor
        for( const CDescriptorBRIEF& cDescriptor: cCloud.second )
        {
            //ds print the descriptor elements
            for( int32_t u = 0; u < DESCRIPTOR_SIZE_BITS; ++u ){ CLogger::writeDatum( ofCloud, cDescriptor[u] ); }
        }
    }
    ofCloud.close( );

    //ds store clouds number
    CLogger::writeDatum( ofCloudQuery, vecCloudsQuery.size( ) );

    //ds save queries
    for( const std::pair< uint64_t, CCloudDescriptorBRIEF> cCloudQuery: vecCloudsQuery )
    {
        //ds write cloud id
        CLogger::writeDatum( ofCloudQuery, cCloudQuery.first );

        //ds store number of descriptors
        CLogger::writeDatum( ofCloudQuery, cCloudQuery.second.size( ) );

        //ds each descriptor
        for( const CDescriptorBRIEF& cDescriptor: cCloudQuery.second )
        {
            //ds print the descriptor elements
            for( int32_t u = 0; u < DESCRIPTOR_SIZE_BITS; ++u ){ CLogger::writeDatum( ofCloudQuery, cDescriptor[u] ); }
        }
    }
    ofCloudQuery.close( );

    std::printf( "(main) clouds creation completed\n" );
    std::fflush( stdout);
    return 0;
}
