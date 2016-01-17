#include "../types/CBRIEFNode.h"
#include "../types/CKeyFrame.h"
#include "../utility/CLogger.h"
#include "../utility/CTimer.h"
#include "../utility/CWrapperOpenCV.h"



#define MAXIMUM_DISTANCE_HAMMING 25
#define MAXIMUM_DEPTH_TREE 8



int32_t main( int32_t argc, char **argv )
{
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
    CLogger::closeBox( );



    std::printf( "(main) starting BINARY TREE matching\n" );
    for( const CKeyFrame* pCloudQuery: vecClouds )
    {
        //ds stop time
        const double dTimeStartSeconds = CLogger::getTimeSeconds( );
        uint64_t uTotalMatches = 0;

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
                            ++uTotalMatches;
                            break;
                        }
                    }
                    break;
                }
            }
        }

        std::printf( "(main) completed matching for clouds %06lu > %06lu matches: %4.2f duration: %f\n", pCloudQuery->uID, pCloudReference->uID, static_cast< double >( uTotalMatches )/pCloudQuery->vecDescriptorPool.size( ), CTimer::getTimeSeconds( )-dTimeStartSeconds );
    }


    std::printf( "\n(main) starting BRUTEFORCE matching\n" );

    //ds for all descriptors in the query clouds
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

        std::printf( "(main) completed matching for clouds %06lu > %06lu matches: %4.2f duration: %f\n", pCloudQuery->uID, pCloudReference->uID, static_cast< double >( uTotalMatches )/pCloudQuery->vecDescriptorPool.size( ), CTimer::getTimeSeconds( )-dTimeStartSeconds );
    }



    delete pRootReference;
    std::fflush( stdout);
    return 0;
}
