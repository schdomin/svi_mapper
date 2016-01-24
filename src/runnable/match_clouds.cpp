#include "../types/CBTree.h"
#include "../types/CKeyFrame.h"
#include "../utility/CLogger.h"
#include "../utility/CTimer.h"
#include "../utility/CWrapperOpenCV.h"



#define MAXIMUM_DISTANCE_HAMMING 25
#define MAXIMUM_DEPTH_TREE 100
#define NUMBER_OF_SAMPLES 1



int32_t main( int32_t argc, char **argv )
{
    assert( false );

    //ds if no clouds provided (at least two)
    if( 3 > argc )
    {
        std::printf( "(main) insufficient amount of clouds (provide at least 2)\n" );
        std::fflush( stdout);
        return 0;
    }

    //ds log file
    //std::FILE* ofResults = std::fopen( "/home/dom/Documents/binary_tree/benchmark_dense.txt", "w" );

    //ds box import
    CLogger::openBox( );

    //ds clouds
    std::vector< CKeyFrame* > vecClouds;

    //ds for each cloud
    for( int32_t u = 0; u < argc-1; ++u )
    {
        //ds buffer to string
        const std::string strCloudfile( argv[u+1] );

        std::printf( "(main) loading cloud: %s - ", argv[vecClouds.size( )+1] );

        try
        {
            //ds try to create a keyframe with the cloud
            vecClouds.push_back( new CKeyFrame( strCloudfile ) );
            std::printf( "points: %lu\n", vecClouds.back( )->vecCloud->size( ) );
        }
        catch( std::exception& p_cException )
        {
            //ds failed to load
            std::printf( "unable to load cloud, exception: '%s'\n", p_cException.what( ) );
        }
    }

    std::printf( "(main) successfully loaded %lu clouds\n", vecClouds.size( ) );
    std::printf( "(main) reference cloud set as: '%s'\n", argv[argc-1] );

    //ds buffer reference cloud (last element)
    const CKeyFrame* pCloudReference = vecClouds.back( );
    vecClouds.pop_back( );



    //ds grow reference tree
    const CBTree< MAXIMUM_DEPTH_TREE, DESCRIPTOR_SIZE_BITS > cBTree( pCloudReference->vecDescriptorPool );

    //ds allocate a flann based matchers
    cv::setNumThreads( 0 );
    cv::FlannBasedMatcher cMatcher0( new cv::flann::LshIndexParams( 1, 20, 0 ) );
    cv::FlannBasedMatcher cMatcher1( new cv::flann::LshIndexParams( 1, 20, 1 ) );
    CLogger::closeBox( );


    CLogger::openBox( );
    //std::fprintf( ofResults, "#cloud #btree le #btree #flann lsh1 #flann lsh0 #bruteforce\n" );
    std::printf( "(main) starting sampling [%u]\n", NUMBER_OF_SAMPLES );
    for( const CKeyFrame* pCloudQuery: vecClouds )
    {
        //ds sampling setup
        uint64_t uTotalMatchesBINARY = 0;
        uint64_t uTotalMatchesBINARYLE = 0;
        uint64_t uTotalMatchesFLANN  = 0;
        uint64_t uTotalMatchesFLANN1 = 0;
        //uint64_t uTotalMatchesBRUTE  = 0;
        std::vector< double > vecDurationSecondsBINARY( NUMBER_OF_SAMPLES, 0.0 );
        std::vector< double > vecDurationSecondsBINARYLE( NUMBER_OF_SAMPLES, 0.0 );
        std::vector< double > vecDurationSecondsFLANN( NUMBER_OF_SAMPLES, 0.0 );
        std::vector< double > vecDurationSecondsFLANN1( NUMBER_OF_SAMPLES, 0.0 );
        //std::vector< double > vecDurationSecondsBRUTE( NUMBER_OF_SAMPLES, 0.0 );

        //ds do samples
        for( uint32_t uSample = 0; uSample < NUMBER_OF_SAMPLES; ++ uSample )
        {
            //ds binary tree sample
            const double dTimeStartSecondsBINARY = CLogger::getTimeSeconds( );
            uTotalMatchesBINARY = cBTree.getNumberOfMatches( pCloudQuery->vecDescriptorPool, MAXIMUM_DISTANCE_HAMMING );
            vecDurationSecondsBINARY[uSample] = CTimer::getTimeSeconds( )-dTimeStartSecondsBINARY;

            //ds binary tree lazy evaluation
            uTotalMatchesBINARYLE = 0;
            const double dTimeStartSecondsBINARYLE = CLogger::getTimeSeconds( );
            uTotalMatchesBINARYLE = cBTree.getNumberOfMatchesFirst( pCloudQuery->vecDescriptorPool, MAXIMUM_DISTANCE_HAMMING );
            vecDurationSecondsBINARYLE[uSample] = CTimer::getTimeSeconds( )-dTimeStartSecondsBINARYLE;



            //ds----------------------------------------------------------------------------- FLANN LSH
            assert( static_cast< std::vector< CDescriptorBRIEF >::size_type >( pCloudQuery->vecDescriptorPoolCV.rows ) == pCloudQuery->vecDescriptorPool.size( ) );

            //ds reset current sample
            uTotalMatchesFLANN = 0;
            const double dTimeStartSecondsFLANN = CLogger::getTimeSeconds( );

            std::vector< cv::DMatch > vecMatches;
            cMatcher0.match( pCloudQuery->vecDescriptorPoolCV, pCloudReference->vecDescriptorPoolCV, vecMatches );

            //ds check matches
            for( const cv::DMatch& cMatch: vecMatches )
            {
                if( MAXIMUM_DISTANCE_HAMMING > cMatch.distance )
                {
                    ++uTotalMatchesFLANN;
                }
            }
            vecDurationSecondsFLANN[uSample] = CTimer::getTimeSeconds( )-dTimeStartSecondsFLANN;
            //ds-----------------------------------------------------------------------------



            //ds----------------------------------------------------------------------------- FLANN LSH MULTIPROBE 1
            assert( static_cast< std::vector< CDescriptorBRIEF >::size_type >( pCloudQuery->vecDescriptorPoolCV.rows ) == pCloudQuery->vecDescriptorPool.size( ) );

            //ds reset current sample
            uTotalMatchesFLANN1 = 0;
            const double dTimeStartSecondsFLANN1 = CLogger::getTimeSeconds( );

            std::vector< cv::DMatch > vecMatches1;
            cMatcher1.match( pCloudQuery->vecDescriptorPoolCV, pCloudReference->vecDescriptorPoolCV, vecMatches1 );

            //ds check matches
            for( const cv::DMatch& cMatch: vecMatches1 )
            {
                if( MAXIMUM_DISTANCE_HAMMING > cMatch.distance )
                {
                    ++uTotalMatchesFLANN1;
                }
            }
            vecDurationSecondsFLANN1[uSample] = CTimer::getTimeSeconds( )-dTimeStartSecondsFLANN1;
            //ds-----------------------------------------------------------------------------



            /*ds----------------------------------------------------------------------------- BRUTEFORCE
            //ds reset current sample
            uTotalMatchesBRUTE = 0;
            const double dTimeStartSecondsBRUTE = CLogger::getTimeSeconds( );

            for( const CDescriptorBRIEF& cDescriptorQuery: pCloudQuery->vecDescriptorPool )
            {
                //ds match against each of the reference
                for( const CDescriptorBRIEF& cDescriptorReference: pCloudReference->vecDescriptorPool )
                {
                    if( MAXIMUM_DISTANCE_HAMMING > CWrapperOpenCV::getDistanceHamming( cDescriptorQuery, cDescriptorReference ) )
                    {
                        ++uTotalMatchesBRUTE;
                        break;
                    }
                }
            }
            vecDurationSecondsBRUTE[uSample] = CTimer::getTimeSeconds( )-dTimeStartSecondsBRUTE;
            //ds-----------------------------------------------------------------------------*/
        }

        const double dRelativeMatchesBINARY = static_cast< double >( uTotalMatchesBINARY )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSecondsBINARY = ( std::accumulate( vecDurationSecondsBINARY.begin( ), vecDurationSecondsBINARY.end( ), 0.0 ) )/vecDurationSecondsBINARY.size( );
        const double dRelativeMatchesBINARYLE = static_cast< double >( uTotalMatchesBINARYLE )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSecondsBINARYLE = ( std::accumulate( vecDurationSecondsBINARYLE.begin( ), vecDurationSecondsBINARYLE.end( ), 0.0 ) )/vecDurationSecondsBINARYLE.size( );
        const double dRelativeMatchesFLANN  = static_cast< double >( uTotalMatchesFLANN )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSecondsFLANN  = ( std::accumulate( vecDurationSecondsFLANN.begin( ), vecDurationSecondsFLANN.end( ), 0.0 ) )/vecDurationSecondsFLANN.size( );
        const double dRelativeMatchesFLANN1  = static_cast< double >( uTotalMatchesFLANN1 )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSecondsFLANN1  = ( std::accumulate( vecDurationSecondsFLANN1.begin( ), vecDurationSecondsFLANN1.end( ), 0.0 ) )/vecDurationSecondsFLANN1.size( );
        //const double dRelativeMatchesBRUTE  = static_cast< double >( uTotalMatchesBRUTE )/pCloudQuery->vecDescriptorPool.size( );
        //const double dDurationSecondsBRUTE  = ( std::accumulate( vecDurationSecondsBRUTE.begin( ), vecDurationSecondsBRUTE.end( ), 0.0 ) )/vecDurationSecondsBRUTE.size( );

        std::printf( "(main) completed matching for clouds %06lu > %06lu [  BTREE LE] matches: %4.2f duration: %f score: %f\n", pCloudQuery->uID, pCloudReference->uID, dRelativeMatchesBINARYLE, dDurationSecondsBINARYLE, dRelativeMatchesBINARYLE/dDurationSecondsBINARYLE );
        std::printf( "(main)                                               [     BTREE] matches: %4.2f duration: %f score: %f\n", dRelativeMatchesBINARY, dDurationSecondsBINARY, dRelativeMatchesBINARY/dDurationSecondsBINARY );
        std::printf( "(main)                                               [FLANN LSH0] matches: %4.2f duration: %f score: %f\n", dRelativeMatchesFLANN, dDurationSecondsFLANN, dRelativeMatchesFLANN/dDurationSecondsFLANN );
        std::printf( "(main)                                               [FLANN LSH1] matches: %4.2f duration: %f score: %f\n", dRelativeMatchesFLANN1, dDurationSecondsFLANN1, dRelativeMatchesFLANN1/dDurationSecondsFLANN1 );
        //std::printf( "(main)                                               [BRUTEFORCE] matches: %4.2f duration: %f score: %f\n", dRelativeMatchesBRUTE, dDurationSecondsBRUTE, dRelativeMatchesBRUTE/dDurationSecondsBRUTE );

        /*ds write to file stream
        std::fprintf( ofResults, "%03lu %f %f %f %f %f\n", pCloudQuery->uID, dRelativeMatchesBINARYLE/dDurationSecondsBINARYLE,
                                                                             dRelativeMatchesBINARY/dDurationSecondsBINARY,
                                                                             dRelativeMatchesFLANN/dDurationSecondsFLANN,
                                                                             dRelativeMatchesFLANN1/dDurationSecondsFLANN1,
                                                                             dRelativeMatchesBRUTE/dDurationSecondsBRUTE );*/
    }
    //delete pRootReference;
    //std::fclose( ofResults );
    CLogger::closeBox( );


/*
    CLogger::openBox( );
    std::printf( "(main) starting FLANN LSH matching\n" );
    for( const CKeyFrame* pCloudQuery: vecClouds )
    {
        assert( static_cast< std::vector< CDescriptorBRIEF >::size_type >( pCloudQuery->vecDescriptorPoolCV.rows ) == pCloudQuery->vecDescriptorPool.size( ) );

        //ds stop time
        const double dTimeStartSeconds = CLogger::getTimeSeconds( );
        uint64_t uTotalMatches = 0;

        std::vector< cv::DMatch > vecMatches;
        cMatcher.match( pCloudQuery->vecDescriptorPoolCV, pCloudReference->vecDescriptorPoolCV, vecMatches );

        //ds check matches
        for( const cv::DMatch& cMatch: vecMatches )
        {
            if( MAXIMUM_DISTANCE_HAMMING > cMatch.distance )
            {
                ++uTotalMatches;
            }
        }

        const double dRelativeMatches = static_cast< double >( uTotalMatches )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSeconds = CTimer::getTimeSeconds( )-dTimeStartSeconds;
        std::printf( "(main)[FLANN LSH] completed matching for clouds %06lu > %06lu matches: %4.2f duration: %f ratio: %f\n", pCloudQuery->uID, pCloudReference->uID, dRelativeMatches, dDurationSeconds, dRelativeMatches/dDurationSeconds );
    }
    CLogger::closeBox( );



    CLogger::openBox( );
    std::printf( "(main) starting BRUTEFORCE matching\n" );
    for( const CKeyFrame* pCloudQuery: vecClouds )
    {
        //ds stop time
        const double dTimeStartSeconds = CLogger::getTimeSeconds( );
        uint64_t uTotalMatches = 0;

        //ds for each descriptor
        for( const CDescriptorBRIEF& cDescriptorQuery: pCloudQuery->vecDescriptorPool )
        {
            //ds match against each of the reference
            for( const CDescriptorBRIEF& cDescriptorReference: pCloudReference->vecDescriptorPool )
            {
                if( MAXIMUM_DISTANCE_HAMMING > CWrapperOpenCV::getDistanceHamming( cDescriptorQuery, cDescriptorReference ) )
                {
                    ++uTotalMatches;
                    break;
                }
            }
        }

        const double dRelativeMatches = static_cast< double >( uTotalMatches )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSeconds = CTimer::getTimeSeconds( )-dTimeStartSeconds;
        std::printf( "(main)[BRUTEFORCE] completed matching for clouds %06lu > %06lu matches: %4.2f duration: %f ratio: %f\n", pCloudQuery->uID, pCloudReference->uID, dRelativeMatches, dDurationSeconds, dRelativeMatches/dDurationSeconds );
    }
    CLogger::closeBox( );
*/


    //ds finish up
    std::fflush( stdout);
    return 0;
}
