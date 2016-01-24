#include "../types/CBNode.h"
#include "../types/CKeyFrame.h"
#include "../utility/CLogger.h"
#include "../utility/CTimer.h"
#include "../utility/CWrapperOpenCV.h"



#define MAXIMUM_DISTANCE_HAMMING 25
#define MAXIMUM_DEPTH_TREE 50



int32_t main( int32_t argc, char **argv )
{
    assert( false );

    //ds if no clouds provided (exactly two extra)
    if( 3 != argc )
    {
        std::printf( "(main) invalid command line parameters: <clouds_total> <clouds_query>\n" );
        std::fflush( stdout);
        return 0;
    }

    //ds log file
    //std::FILE* ofResults = std::fopen( "/home/dom/Documents/binary_tree/benchmark_dense.txt", "w" );

    //ds box import
    CLogger::openBox( );

    //ds total descriptor counts
    uint64_t uNumberOfDescriptorsTotal = 0;
    uint64_t uNumberOfDescriptorsQuery = 0;

    //ds load clouds
    std::printf( "(main) loading clouds total: '%s'\n", argv[1] );
    std::ifstream ifCloudTotal( argv[1], std::ifstream::in );

    //ds parse number of clouds
    std::vector< std::pair< uint64_t, CCloudDescriptorBRIEF > >::size_type uNumberOfCloudsTotal = 0;
    CLogger::readDatum( ifCloudTotal, uNumberOfCloudsTotal );

    //ds allocate cloud holder
    std::vector< std::pair< uint64_t, CCloudDescriptorBRIEF > > vecCloudsTotal( uNumberOfCloudsTotal );

    //ds parse data
    for( std::vector< std::pair< uint64_t, CCloudDescriptorBRIEF > >::size_type u = 0; u < uNumberOfCloudsTotal; ++u )
    {
        //ds retrieve current cloud id
        uint64_t uID = 0;
        CLogger::readDatum( ifCloudTotal, uID );

        //ds retrieve number of descriptors
        CCloudDescriptorBRIEF::size_type uNumberOfDescriptors = 0;
        CLogger::readDatum( ifCloudTotal, uNumberOfDescriptors );

        //ds allocate local cloud
        CCloudDescriptorBRIEF vecCloud( uNumberOfDescriptors );
        uNumberOfDescriptorsTotal += uNumberOfDescriptors;

        //ds load descriptors
        for( CCloudDescriptorBRIEF::size_type v = 0; v < uNumberOfDescriptors; ++v )
        {
            //ds retrieve descriptor
            CDescriptorBRIEF cDescriptor;
            for( uint32_t w = 0; w < DESCRIPTOR_SIZE_BITS; ++w ){ CLogger::readDatum( ifCloudTotal, cDescriptor[w] ); }

            //ds set the descriptor
            vecCloud[v] = cDescriptor;
        }

        //ds set cloud
        vecCloudsTotal[u] = std::make_pair( uID, vecCloud );
    }
    ifCloudTotal.close( );
    std::printf( "(main) loaded clouds total: %lu (descriptors: %lu)\n", vecCloudsTotal.size( ), uNumberOfDescriptorsTotal );

    //ds load clouds
    std::printf( "(main) loading clouds query: '%s'\n", argv[2] );
    std::ifstream ifCloudQuery( argv[2], std::ifstream::in );

    //ds parse number of clouds
    std::vector< std::pair< uint64_t, CCloudDescriptorBRIEF > >::size_type uNumberOfCloudsQuery = 0;
    CLogger::readDatum( ifCloudQuery, uNumberOfCloudsQuery );

    //ds allocate cloud holder
    std::vector< std::pair< uint64_t, CCloudDescriptorBRIEF > > vecCloudsQuery( uNumberOfCloudsQuery );

    //ds parse data
    for( std::vector< std::pair< uint64_t, CCloudDescriptorBRIEF > >::size_type u = 0; u < uNumberOfCloudsQuery; ++u )
    {
        //ds retrieve current cloud id
        uint64_t uID = 0;
        CLogger::readDatum( ifCloudQuery, uID );

        //ds retrieve number of descriptors
        CCloudDescriptorBRIEF::size_type uNumberOfDescriptors = 0;
        CLogger::readDatum( ifCloudQuery, uNumberOfDescriptors );

        //ds allocate local cloud
        CCloudDescriptorBRIEF vecCloud( uNumberOfDescriptors );
        uNumberOfDescriptorsQuery += uNumberOfDescriptors;

        //ds load descriptors
        for( CCloudDescriptorBRIEF::size_type v = 0; v < uNumberOfDescriptors; ++v )
        {
            //ds retrieve descriptor
            CDescriptorBRIEF cDescriptor;
            for( uint32_t w = 0; w < DESCRIPTOR_SIZE_BITS; ++w ){ CLogger::readDatum( ifCloudQuery, cDescriptor[w] ); }

            //ds set the descriptor
            vecCloud[v] = cDescriptor;
        }

        //ds set cloud
        vecCloudsQuery[u] = std::make_pair( uID, vecCloud );
    }
    ifCloudQuery.close( );
    std::printf( "(main) loaded clouds query: %lu (descriptors: %lu)\n", vecCloudsQuery.size( ), uNumberOfDescriptorsQuery );


    //ds consts
    const std::vector< std::pair< uint64_t, CCloudDescriptorBRIEF > >::size_type uNumberOfCloudsTotalFinal = vecCloudsTotal.size( );


    //ds loop over references (flip reference/query scheme here)
    for( const std::pair< uint64_t, CCloudDescriptorBRIEF >& prCloudReference: vecCloudsQuery )
    {
        //ds timing
        const double dTimeStartSeconds = CLogger::getTimeSeconds( );

        //ds build tree on query cloud
        const CBNode< MAXIMUM_DEPTH_TREE, DESCRIPTOR_SIZE_BITS >* pRootReference = new CBNode< MAXIMUM_DEPTH_TREE, DESCRIPTOR_SIZE_BITS >( prCloudReference.second );

        //ds results (ID, relative matching)
        std::vector< std::pair< uint64_t, double > > vecMatches( uNumberOfCloudsTotalFinal );

        //ds for all queries
        for( std::vector< std::pair< uint64_t, CCloudDescriptorBRIEF > >::size_type u = 0; u < uNumberOfCloudsTotalFinal; ++u )
        {
            //ds current matches
            uint64_t uTotalMatches = 0;

            //ds match all descriptors
            for( const CDescriptorBRIEF& cDescriptorQuery: vecCloudsTotal[u].second )
            {
                //ds traverse tree to find this descriptor
                const CBNode< MAXIMUM_DEPTH_TREE, DESCRIPTOR_SIZE_BITS >* pNodeCurrent = pRootReference;
                while( pNodeCurrent )
                {
                    //ds if this node has leaves (is splittable)
                    if( pNodeCurrent->bHasLeaves )
                    {
                        //ds check the split bit and go deeper
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
                                ++uTotalMatches;
                                break;
                            }
                        }
                        break;
                    }
                }
            }

            //std::printf( "(main) %06lu > %06lu: %f\n", prCloudReference.first, prCloudQuery.first, static_cast< double >( uTotalMatches )/prCloudQuery.second.size( ) );

            //ds set result
            vecMatches[u] = std::make_pair( vecCloudsTotal[u].first, static_cast< double >( uTotalMatches )/vecCloudsTotal[u].second.size( ) );
        }

        //ds free tree
        delete pRootReference;

        //ds sort results
        std::sort( vecMatches.begin( ), vecMatches.end( ), []( const std::pair< uint64_t, double >& LHS, const std::pair< uint64_t, double >& RHS ){ return LHS.second > RHS.second; } );

        std::printf( "(main) finished query: %06lu - duration: %fs\n", prCloudReference.first, CTimer::getTimeSeconds( )-dTimeStartSeconds );
        std::printf( "(main) top results: " );
        for( uint8_t u = 0; u < 10; ++u )
        {
            std::printf( "> %06lu ", vecMatches[u].first );
        }
        std::printf( "\n" );
    }





    CLogger::openBox( );
    //std::fprintf( ofResults, "#cloud #btree le #btree #flann lsh1 #flann lsh0 #bruteforce\n" );
    /*for( const CKeyFrame* pCloudQuery: vecClouds )
    {
        //ds sampling setup
        uint64_t uTotalMatchesBINARY = 0;
        uint64_t uTotalMatchesBINARYLE = 0;
        uint64_t uTotalMatchesFLANN  = 0;
        uint64_t uTotalMatchesFLANN1 = 0;
        uint64_t uTotalMatchesBRUTE  = 0;
        std::vector< double > vecDurationSecondsBINARY( NUMBER_OF_SAMPLES, 0.0 );
        std::vector< double > vecDurationSecondsBINARYLE( NUMBER_OF_SAMPLES, 0.0 );
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
                        //ds check the split bit and go deeper
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



            //ds----------------------------------------------------------------------------- BINARY TREE LAZY EVALUATION
            //ds reset current sample
            uTotalMatchesBINARYLE = 0;
            const double dTimeStartSecondsBINARYLE = CLogger::getTimeSeconds( );

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
                        //ds check the split bit and go deeper
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
                        //ds check current descriptor in this node and exit
                        if( MAXIMUM_DISTANCE_HAMMING > CWrapperOpenCV::getDistanceHamming( cDescriptorQuery, pNodeCurrent->vecDescriptors.front( ) ) )
                        {
                            ++uTotalMatchesBINARYLE;
                        }
                        break;
                    }
                }
            }
            vecDurationSecondsBINARYLE[uSample] = CTimer::getTimeSeconds( )-dTimeStartSecondsBINARYLE;
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
            //ds-----------------------------------------------------------------------------*//*
        }

        const double dRelativeMatchesBINARY = static_cast< double >( uTotalMatchesBINARY )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSecondsBINARY = ( std::accumulate( vecDurationSecondsBINARY.begin( ), vecDurationSecondsBINARY.end( ), 0.0 ) )/vecDurationSecondsBINARY.size( );
        const double dRelativeMatchesBINARYLE = static_cast< double >( uTotalMatchesBINARYLE )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSecondsBINARYLE = ( std::accumulate( vecDurationSecondsBINARYLE.begin( ), vecDurationSecondsBINARYLE.end( ), 0.0 ) )/vecDurationSecondsBINARYLE.size( );
        const double dRelativeMatchesFLANN  = static_cast< double >( uTotalMatchesFLANN )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSecondsFLANN  = ( std::accumulate( vecDurationSecondsFLANN.begin( ), vecDurationSecondsFLANN.end( ), 0.0 ) )/vecDurationSecondsFLANN.size( );
        const double dRelativeMatchesFLANN1  = static_cast< double >( uTotalMatchesFLANN1 )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSecondsFLANN1  = ( std::accumulate( vecDurationSecondsFLANN1.begin( ), vecDurationSecondsFLANN1.end( ), 0.0 ) )/vecDurationSecondsFLANN1.size( );
        const double dRelativeMatchesBRUTE  = static_cast< double >( uTotalMatchesBRUTE )/pCloudQuery->vecDescriptorPool.size( );
        const double dDurationSecondsBRUTE  = ( std::accumulate( vecDurationSecondsBRUTE.begin( ), vecDurationSecondsBRUTE.end( ), 0.0 ) )/vecDurationSecondsBRUTE.size( );

        std::printf( "(main) completed matching for clouds %06lu > %06lu [  BTREE LE] matches: %4.2f duration: %f score: %f\n", pCloudQuery->uID, pCloudReference->uID, dRelativeMatchesBINARYLE, dDurationSecondsBINARYLE, dRelativeMatchesBINARYLE/dDurationSecondsBINARYLE );
        std::printf( "(main)                                               [     BTREE] matches: %4.2f duration: %f score: %f\n", dRelativeMatchesBINARY, dDurationSecondsBINARY, dRelativeMatchesBINARY/dDurationSecondsBINARY );
        std::printf( "(main)                                               [FLANN LSH0] matches: %4.2f duration: %f score: %f\n", dRelativeMatchesFLANN, dDurationSecondsFLANN, dRelativeMatchesFLANN/dDurationSecondsFLANN );
        std::printf( "(main)                                               [FLANN LSH1] matches: %4.2f duration: %f score: %f\n", dRelativeMatchesFLANN1, dDurationSecondsFLANN1, dRelativeMatchesFLANN1/dDurationSecondsFLANN1 );
        std::printf( "(main)                                               [BRUTEFORCE] matches: %4.2f duration: %f score: %f\n", dRelativeMatchesBRUTE, dDurationSecondsBRUTE, dRelativeMatchesBRUTE/dDurationSecondsBRUTE );

        //ds write to file stream
        std::fprintf( ofResults, "%03lu %f %f %f %f %f\n", pCloudQuery->uID, dRelativeMatchesBINARYLE/dDurationSecondsBINARYLE,
                                                                             dRelativeMatchesBINARY/dDurationSecondsBINARY,
                                                                             dRelativeMatchesFLANN/dDurationSecondsFLANN,
                                                                             dRelativeMatchesFLANN1/dDurationSecondsFLANN1,
                                                                             dRelativeMatchesBRUTE/dDurationSecondsBRUTE );
    }*/
    //std::fclose( ofResults );
    CLogger::closeBox( );

    //ds finish up
    std::fflush( stdout);
    return 0;
}
