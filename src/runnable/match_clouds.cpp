#include "../types/CBRIEFNode.h"
#include "../types/CKeyFrame.h"
#include "../utility/CLogger.h"
#include "../utility/CTimer.h"
#include "../utility/CWrapperOpenCV.h"



#define MAXIMUM_DISTANCE_HAMMING 25
#define MAXIMUM_DEPTH_TREE 20
#define NUMBER_OF_SAMPLES 10



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

    //ds buffer reference cloud (last element)
    const CKeyFrame* pCloudReference = vecClouds.back( );
    vecClouds.pop_back( );



    //ds reference tree
    std::printf( "(main) loading tree [%u|%u]\n", MAXIMUM_DEPTH_TREE, DESCRIPTOR_SIZE_BITS );
    const CBRIEFNode< MAXIMUM_DEPTH_TREE, DESCRIPTOR_SIZE_BITS >* pRootReference = new CBRIEFNode< MAXIMUM_DEPTH_TREE, DESCRIPTOR_SIZE_BITS >( 0, pCloudReference->vecDescriptorPool );
    std::printf( "(main) tree successfully loaded\n" );

    //ds allocate a flann based matchers
    cv::FlannBasedMatcher cMatcher0( new cv::flann::LshIndexParams( 1, 20, 0 ) );
    cv::FlannBasedMatcher cMatcher1( new cv::flann::LshIndexParams( 1, 20, 1 ) );
    CLogger::closeBox( );


    CLogger::openBox( );
    std::printf( "(main) starting sampling [%u]\n", NUMBER_OF_SAMPLES );
    for( const CKeyFrame* pCloudQuery: vecClouds )
    {
        //ds sampling setup
        uint64_t uTotalMatchesBINARY = 0;
        uint64_t uTotalMatchesFLANN  = 0;
        uint64_t uTotalMatchesFLANN1 = 0;
        uint64_t uTotalMatchesBRUTE  = 0;
        std::vector< double > vecDurationSecondsBINARY( NUMBER_OF_SAMPLES, 0.0 );
        std::vector< double > vecDurationSecondsFLANN( NUMBER_OF_SAMPLES, 0.0 );
        std::vector< double > vecDurationSecondsFLANN1( NUMBER_OF_SAMPLES, 0.0 );
        std::vector< double > vecDurationSecondsBRUTE( NUMBER_OF_SAMPLES, 0.0 );

        //ds do samples
        for( uint32_t uSample = 0; uSample < NUMBER_OF_SAMPLES; ++ uSample )
        {
            //ds----------------------------------------------------------------------------- BINARY TREE
            //ds reset current sample
            uTotalMatchesBINARY = 0;
            const double dTimeStartSecondsBINARY = CLogger::getTimeSeconds( );

            //ds for each descriptor
            for( const CDescriptorBRIEF& cDescriptorQuery: pCloudQuery->vecDescriptorPool )
            {
                //ds traverse tree to find this descriptor
                const CBRIEFNode< MAXIMUM_DEPTH_TREE, DESCRIPTOR_SIZE_BITS >* pNodeCurrent = pRootReference;
                while( pNodeCurrent )
                {
                    //ds if this node has leaves (is splittable)
                    if( pNodeCurrent->bHasLeaves )
                    {
                        //ds check the split bit and go deeper if set
                        if( cDescriptorQuery[pNodeCurrent->uIndexSplitBit] )
                        {
                            pNodeCurrent = pNodeCurrent->pLeafOnes;
                        }
                        else
                        {
                            pNodeCurrent = pNodeCurrent->pLeafZeros;
                        }
                    }
                    else
                    {
                        //ds check current descriptors in this node and exit
                        for( const CDescriptorBRIEF& cDescriptorReference: pNodeCurrent->vecDescriptors )
                        {
                            if( MAXIMUM_DISTANCE_HAMMING > CWrapperOpenCV::getDistanceHamming( cDescriptorQuery, cDescriptorReference ) )
                            {
                                ++uTotalMatchesBINARY;
                                break;
                            }
                        }
                        break;
                    }
                }
            }
            vecDurationSecondsBINARY[uSample] = CTimer::getTimeSeconds( )-dTimeStartSecondsBINARY;
            //ds-----------------------------------------------------------------------------




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



            //ds----------------------------------------------------------------------------- BRUTEFORCE
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
            //ds-----------------------------------------------------------------------------
        }

        const double dRelativeMatchesBINARY = static_cast< double >( uTotalMatchesBINARY )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSecondsBINARY = ( std::accumulate( vecDurationSecondsBINARY.begin( ), vecDurationSecondsBINARY.end( ), 0.0 ) )/vecDurationSecondsBINARY.size( );
        const double dRelativeMatchesFLANN  = static_cast< double >( uTotalMatchesFLANN )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSecondsFLANN  = ( std::accumulate( vecDurationSecondsFLANN.begin( ), vecDurationSecondsFLANN.end( ), 0.0 ) )/vecDurationSecondsFLANN.size( );
        const double dRelativeMatchesFLANN1  = static_cast< double >( uTotalMatchesFLANN1 )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSecondsFLANN1  = ( std::accumulate( vecDurationSecondsFLANN1.begin( ), vecDurationSecondsFLANN1.end( ), 0.0 ) )/vecDurationSecondsFLANN1.size( );
        const double dRelativeMatchesBRUTE  = static_cast< double >( uTotalMatchesBRUTE )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSecondsBRUTE  = ( std::accumulate( vecDurationSecondsBRUTE.begin( ), vecDurationSecondsBRUTE.end( ), 0.0 ) )/vecDurationSecondsBRUTE.size( );

        std::printf( "(main) completed matching for clouds %06lu > %06lu [BB TREE   ] matches: %4.2f duration: %f ratio: %f\n", pCloudQuery->uID, pCloudReference->uID, dRelativeMatchesBINARY, dDurationSecondsBINARY, dRelativeMatchesBINARY/dDurationSecondsBINARY );
        std::printf( "(main)                                               [FLANN LSH0] matches: %4.2f duration: %f ratio: %f\n", dRelativeMatchesFLANN, dDurationSecondsFLANN, dRelativeMatchesFLANN/dDurationSecondsFLANN );
        std::printf( "(main)                                               [FLANN LSH1] matches: %4.2f duration: %f ratio: %f\n", dRelativeMatchesFLANN1, dDurationSecondsFLANN1, dRelativeMatchesFLANN1/dDurationSecondsFLANN1 );
        std::printf( "(main)                                               [BRUTEFORCE] matches: %4.2f duration: %f ratio: %f\n", dRelativeMatchesBRUTE, dDurationSecondsBRUTE, dRelativeMatchesBRUTE/dDurationSecondsBRUTE );
    }
    delete pRootReference;
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
