#ifndef CBITREE_H
#define CBITREE_H

#include "CBNode.h"



template< uint32_t uMaximumDistanceHamming = 25, uint64_t uMaximumDepth = 50, uint32_t uDescriptorSizeBits = 256 >
class CBITree
{

//ds ctor/dtor
public:

    //ds construct empty tree upon allocation
    CBITree( ): m_pRoot( 0 )
    {
#if defined REBUILD_BITREE

        m_vecTotalDescriptors.clear( );
        m_mapBitStatistics.clear( );
        m_vecNewDescriptorsBuffer.clear( );

#endif
    }

    //ds free all nodes in the tree
    ~CBITree( )
    {
        //ds erase all nodes
        displant( );
    }

private:

    CBNode< uMaximumDepth, uDescriptorSizeBits >* m_pRoot;

#if defined REBUILD_BITREE

    std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > m_vecTotalDescriptors;
    std::map< UIDLandmark, CBitStatistics > m_mapBitStatistics;
    std::vector< std::pair< std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >, std::map< UIDLandmark, CBitStatistics > > > m_vecNewDescriptorsBuffer;
    const uint32_t uMiniumBatchSizeForRebuild = 0;

#else

    uint64_t m_uTotalNumberOfDescriptors = 0;

#endif

//ds access
public:

    void add( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptorsNEW )
    {
        assert( 0 < p_vecDescriptorsNEW.size( ) );

        //ds get filtered descriptors
        const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsNEWFiltered( CBNode< uMaximumDepth, uDescriptorSizeBits >::getFilteredDescriptorsExhaustive( p_vecDescriptorsNEW ) );

#if defined REBUILD_BITREE

        //ds displant current tree
        displant( );

        //ds add the descriptors to the complete pool
        for( const CDescriptorBRIEF< uDescriptorSizeBits >& p_cDescriptor: vecDescriptorsNEWFiltered )
        {
            m_vecTotalDescriptors.push_back( p_cDescriptor );
        }

        //ds grow new tree on descriptors
        m_pRoot = new CBNode< uMaximumDepth, uDescriptorSizeBits >( m_vecTotalDescriptors );

        //ds tree stats
        uint64_t uDepth            = 0;
        uint64_t uNumberOfEndNodes = 0;
        _setInfoRecursive( m_pRoot, uDepth, uNumberOfEndNodes );

        //ds log aggregation
        std::ofstream ofLogFile( "logs/growth_rbitree.txt", std::ofstream::out | std::ofstream::app );
        ofLogFile << vecDescriptorsNEWFiltered.front( ).uIDKeyFrame
                  << " " << vecDescriptorsNEWFiltered.size( )
                  << " " << 0
                  << " " << uDepth
                  << " " << static_cast< double >( m_vecTotalDescriptors.size( ) )/uNumberOfEndNodes
                  << " " << m_vecTotalDescriptors.size( )
                  << " " << uNumberOfEndNodes << "\n";
        ofLogFile.close( );

#else

        //ds info
        uint64_t uNumberOfNonAggregations = 0;

        //ds if the tree is set
        if( m_pRoot )
        {
            //ds for each new descriptor
            for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorNEW: p_vecDescriptorsNEW )
            {
                //ds success flag
                bool bFailedToInsertDescriptor = true;

                //ds traverse tree to find a leaf for this descriptor
                CBNode< uMaximumDepth, uDescriptorSizeBits >* pNodeCurrent = m_pRoot;
                while( pNodeCurrent )
                {
                    //ds if this node has leaves (is splittable)
                    if( pNodeCurrent->bHasLeaves )
                    {
                        //ds check the split bit and go deeper
                        if( cDescriptorNEW.vecData[pNodeCurrent->uIndexSplitBit] )
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
                        //ds we ended up in a final leaf - lets check if our descriptor matches the criteria to remain
                        for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorTRAIN: pNodeCurrent->vecDescriptors )
                        {
                            if( uMaximumDistanceHamming > CBNode< uMaximumDepth, uDescriptorSizeBits >::getDistanceHamming( cDescriptorNEW.vecData, cDescriptorTRAIN.vecData ) )
                            {
                                //ds place the descriptor into the leaf
                                pNodeCurrent->vecDescriptors.push_back( cDescriptorNEW );

                                //ds done
                                bFailedToInsertDescriptor = false;
                                break;
                            }
                        }

                        //ds check if we couldn't insert the descriptor
                        if( bFailedToInsertDescriptor )
                        {
                            //ds place the descriptor into the leaf
                            pNodeCurrent->vecDescriptors.push_back( cDescriptorNEW );

                            //ds try to split the leaf
                            if( pNodeCurrent->spawnLeafs( ) )
                            {
                                //ds success
                                assert( pNodeCurrent->bHasLeaves );
                                bFailedToInsertDescriptor = false;
                            }
                        }

                        //ds escape tree for this descriptor
                        break;
                    }
                }

                //ds if failed
                if( bFailedToInsertDescriptor )
                {
                    ++uNumberOfNonAggregations;
                }
            }
        }
        else
        {
            //ds grow initial tree on root
            m_pRoot = new CBNode< uMaximumDepth, uDescriptorSizeBits >( p_vecDescriptorsNEW );
        }

        //ds tree stats
        uint64_t uDepth            = 0;
        uint64_t uNumberOfEndNodes = 0;
        _setInfoRecursive( m_pRoot, uDepth, uNumberOfEndNodes );
        m_uTotalNumberOfDescriptors += ( p_vecDescriptorsNEW.size( )-uNumberOfNonAggregations );

        //ds log aggregation
        std::ofstream ofLogFile( "logs/growth_bitree.txt", std::ofstream::out | std::ofstream::app );
        ofLogFile << p_vecDescriptorsNEW.front( ).uIDKeyFrame
                  << " " << p_vecDescriptorsNEW.size( )
                  << " " << uNumberOfNonAggregations
                  << " " << uDepth
                  << " " << static_cast< double >( m_uTotalNumberOfDescriptors )/uNumberOfEndNodes
                  << " " << m_uTotalNumberOfDescriptors
                  << " " << uNumberOfEndNodes << "\n";
        ofLogFile.close( );

#endif

    }

#if defined REBUILD_BITREE

    void add( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptorsNEW, const std::map< UIDLandmark, CBitStatistics >& p_mapBitStatistics )
    {
        assert( 0 < p_vecDescriptorsNEW.size( ) );

        //ds get filtered descriptors
        const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsNEWFiltered( CBNode< uMaximumDepth, uDescriptorSizeBits >::getFilteredDescriptorsExhaustive( p_vecDescriptorsNEW ) );

        //ds add to buffer
        m_vecNewDescriptorsBuffer.push_back( std::make_pair( vecDescriptorsNEWFiltered, p_mapBitStatistics ) );

        //ds if size is sufficient
        if( uMiniumBatchSizeForRebuild < m_vecNewDescriptorsBuffer.size( ) )
        {
            const double dTimeStartSeconds = CTimer::getTimeSeconds( );

            //ds displant current tree
            displant( );

            uint64_t NumberOfUpdatedLandmarks = 0;
            const uint64_t uInitialNumberOfDescriptors = m_vecTotalDescriptors.size( );

            //ds loop over descriptor pools to add
            for( const std::pair< std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >, std::map< UIDLandmark, CBitStatistics > >& prNewDescriptors: m_vecNewDescriptorsBuffer )
            {
                //ds add the descriptors to the complete pool
                for( const CDescriptorBRIEF< uDescriptorSizeBits >& p_cDescriptor: prNewDescriptors.first )
                {
                    m_vecTotalDescriptors.push_back( p_cDescriptor );
                }

                //ds update statistics
                for( const std::pair< UIDLandmark, CBitStatistics >& prBitStatistics: prNewDescriptors.second )
                {
                    //ds try to insert into existing element
                    std::pair< std::map< UIDLandmark, CBitStatistics >::iterator, bool > prInsertion( m_mapBitStatistics.insert( prBitStatistics ) );

                    if( false == prInsertion.second )
                    {
                        //ds update element
                        prInsertion.first->second = prBitStatistics.second;
                        ++NumberOfUpdatedLandmarks;
                    }
                }
            }

            //ds grow new tree on descriptors
            m_pRoot = new CBNode< uMaximumDepth, uDescriptorSizeBits >( m_vecTotalDescriptors, m_mapBitStatistics );

            //ds tree stats
            uint64_t uDepth            = 0;
            uint64_t uNumberOfEndNodes = 0;
            _setInfoRecursive( m_pRoot, uDepth, uNumberOfEndNodes );

            //ds log aggregation
            std::ofstream ofLogFile( "logs/growth_rbitree.txt", std::ofstream::out | std::ofstream::app );
            ofLogFile << m_vecNewDescriptorsBuffer.front( ).first.front( ).uIDKeyFrame
                      << " " << m_vecTotalDescriptors.size( )-uInitialNumberOfDescriptors
                      << " " << NumberOfUpdatedLandmarks
                      << " " << uDepth
                      << " " << static_cast< double >( m_vecTotalDescriptors.size( ) )/uNumberOfEndNodes
                      << " " << m_vecTotalDescriptors.size( )
                      << " " << uNumberOfEndNodes
                      << " " << CTimer::getTimeSeconds( )-dTimeStartSeconds << "\n";
            ofLogFile.close( );

            m_vecNewDescriptorsBuffer.clear( );
        }
    }

#endif

    void match( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptorsQUERY, const UIDKeyFrame& p_uIDQUERY, std::vector< cv::DMatch >& p_vecMatches ) const
    {
        //ds for each descriptor
        for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorQUERY: p_vecDescriptorsQUERY )
        {
            //ds traverse tree to find this descriptor
            const CBNode< uMaximumDepth, uDescriptorSizeBits >* pNodeCurrent = m_pRoot;
            while( pNodeCurrent )
            {
                //ds if this node has leaves (is splittable)
                if( pNodeCurrent->bHasLeaves )
                {
                    //ds check the split bit and go deeper
                    if( cDescriptorQUERY.vecData[pNodeCurrent->uIndexSplitBit] )
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
                    //ds set of matched ids
                    std::set< UIDKeyFrame > setMatched;

                    //ds check current descriptors in this node and exit
                    for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorTRAIN: pNodeCurrent->vecDescriptors )
                    {
                        if( uMaximumDistanceHamming > CBNode< uMaximumDepth, uDescriptorSizeBits >::getDistanceHamming( cDescriptorQUERY.vecData, cDescriptorTRAIN.vecData ) )
                        {
                            //ds if not matched yet
                            if( 0 == setMatched.count( cDescriptorTRAIN.uIDKeyFrame ) )
                            {
                                //ds add all matches for different key frames - NOT BREAKING
                                p_vecMatches.push_back( cv::DMatch( cDescriptorQUERY.uID, cDescriptorTRAIN.uID, cDescriptorTRAIN.uIDKeyFrame, uMaximumDistanceHamming ) );
                                setMatched.insert( cDescriptorTRAIN.uIDKeyFrame );
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    //ds direct matching function on this tree
    void match( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptorsQUERY, std::vector< cv::DMatch >& p_vecMatches ) const
    {
        //ds for each descriptor
        for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorQUERY: p_vecDescriptorsQUERY )
        {
            //ds traverse tree to find this descriptor
            const CBNode< uMaximumDepth, uDescriptorSizeBits >* pNodeCurrent = m_pRoot;
            while( pNodeCurrent )
            {
                //ds if this node has leaves (is splittable)
                if( pNodeCurrent->bHasLeaves )
                {
                    //ds check the split bit and go deeper
                    if( cDescriptorQUERY.vecData[pNodeCurrent->uIndexSplitBit] )
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
                    for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorTRAIN: pNodeCurrent->vecDescriptors )
                    {
                        if( uMaximumDistanceHamming > CBNode< uMaximumDepth, uDescriptorSizeBits >::getDistanceHamming( cDescriptorQUERY.vecData, cDescriptorTRAIN.vecData ) )
                        {
                            //++pNodeCurrent->uLinkedPoints;
                            p_vecMatches.push_back( cv::DMatch( cDescriptorQUERY.uID, cDescriptorTRAIN.uID, uMaximumDistanceHamming ) );
                            break;
                        }
                    }
                    break;
                }
            }
        }
    }

    //ds delete tree
    void displant( )
    {
        //ds if set
        if( m_pRoot )
        {
            //ds nodes holder
            std::vector< const CBNode< uMaximumDepth, uDescriptorSizeBits >* > vecNodes;

            //ds set vector
            _setNodesRecursive( m_pRoot, vecNodes );

            //ds free nodes
            for( const CBNode< uMaximumDepth, uDescriptorSizeBits >* pNode: vecNodes )
            {
                delete pNode;
            }

            //ds free all nodes
            //std::printf( "(CBITree) deallocated nodes: %lu\n", vecNodes.size( ) );
            vecNodes.clear( );
        }
    }

    //ds constant root node access for reading
    const CBNode< uMaximumDepth, uDescriptorSizeBits >* getRoot( ) const { return m_pRoot; }

//ds helpers
private:

    void _setNodesRecursive( const CBNode< uMaximumDepth, uDescriptorSizeBits >* p_pNode, std::vector< const CBNode< uMaximumDepth, uDescriptorSizeBits >* >& p_vecNodes ) const
    {
        //ds must not be zero
        assert( 0 != p_pNode );

        //ds add the current node
        p_vecNodes.push_back( p_pNode );

        //ds check if there are leafs
        if( p_pNode->bHasLeaves )
        {
            //ds add leafs and so on
            _setNodesRecursive( p_pNode->pLeafOnes, p_vecNodes );
            _setNodesRecursive( p_pNode->pLeafZeros, p_vecNodes );
        }
    }

    void _setEndNodesRecursive( CBNode< uMaximumDepth, uDescriptorSizeBits >* p_pNode, std::vector< CBNode< uMaximumDepth, uDescriptorSizeBits >* >& p_vecNodes )
    {
        //ds must not be zero
        assert( 0 != p_pNode );

        //ds check if there are no leafs
        if( !p_pNode->bHasLeaves )
        {
            //ds add the current node
            p_vecNodes.push_back( p_pNode );
        }
        else
        {
            //ds check leafs
            _setEndNodesRecursive( p_pNode->pLeafOnes, p_vecNodes );
            _setEndNodesRecursive( p_pNode->pLeafZeros, p_vecNodes );
        }
    }

    void _setInfoRecursive( const CBNode< uMaximumDepth, uDescriptorSizeBits >* p_pNode, uint64_t& p_uMaximumDepth, uint64_t& p_uNumberOfEndNodes ) const
    {
        //ds must not be zero
        assert( 0 != p_pNode );

        //ds check if there are no more leafs
        if( !p_pNode->bHasLeaves )
        {
            //ds add the current node
            ++p_uNumberOfEndNodes;

            //ds if depth is higher
            if( p_uMaximumDepth < p_pNode->uDepth )
            {
                //ds update depth
                p_uMaximumDepth = p_pNode->uDepth;
            }
        }
        else
        {
            //ds check leafs
            _setInfoRecursive( p_pNode->pLeafOnes, p_uMaximumDepth, p_uNumberOfEndNodes );
            _setInfoRecursive( p_pNode->pLeafZeros, p_uMaximumDepth, p_uNumberOfEndNodes );
        }
    }

};

#endif //CBITREE_H
