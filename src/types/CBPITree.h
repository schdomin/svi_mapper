#ifndef CBPITREE_H
#define CBPITREE_H

#include "CBPNode.h"



template< uint32_t uMaximumDistanceHammingProbability = 50, uint64_t uMaximumDepth = 50, uint32_t uDescriptorSizeBits = 256 >
class CBPITree
{

//ds ctor/dtor
public:

    //ds construct empty tree upon allocation
    CBPITree( ): m_pRoot( 0 )
    {
        //ds nothing to do
    }

    //ds free all nodes in the tree
    ~CBPITree( )
    {
        //ds erase all nodes
        displant( );
    }

private:

    CBPNode< uMaximumDepth, uDescriptorSizeBits >* m_pRoot;
    uint64_t m_uTotalNumberOfDescriptors = 0;

#if defined REBUILD_BPITREE

    std::vector< CPDescriptorBRIEF< uDescriptorSizeBits > > m_vecTotalDescriptors;

#endif

//ds access
public:

    void add( const std::vector< CPDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptorsNEW )
    {
        assert( 0 < p_vecDescriptorsNEW.size( ) );

        //ds info
        uint64_t uNumberOfNonAggregations = 0;

#if defined REBUILD_BPITREE

        //ds displant current tree
        displant( );

        //ds add the descriptors to the complete pool
        for( const CPDescriptorBRIEF< uDescriptorSizeBits >& p_cDescriptor: p_vecDescriptorsNEW )
        {
            m_vecTotalDescriptors.push_back( p_cDescriptor );
        }

        //ds always grow new tree on descriptors
        m_pRoot = new CBPNode< uMaximumDepth, uDescriptorSizeBits >( m_vecTotalDescriptors );

#else

        //ds if the tree is set
        if( m_pRoot )
        {
            //ds for each new descriptor
            for( const CPDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorNEW: p_vecDescriptorsNEW )
            {
                //ds success flag
                bool bFailedToInsertDescriptor = true;

                //ds traverse tree to find a leaf for this descriptor
                CBPNode< uMaximumDepth, uDescriptorSizeBits >* pNodeCurrent = m_pRoot;
                while( pNodeCurrent )
                {
                    //ds if this node has leaves (is splittable)
                    if( pNodeCurrent->bHasLeaves )
                    {
                        //ds check the split bit and go deeper
                        if( 0.5 < cDescriptorNEW.vecData[pNodeCurrent->uIndexSplitBit] )
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
                        for( const CPDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorTRAIN: pNodeCurrent->vecDescriptors )
                        {
                            if( uMaximumDistanceHammingProbability > CBPNode< uMaximumDepth, uDescriptorSizeBits >::getDistanceHammingProbability( cDescriptorNEW.vecData, cDescriptorTRAIN.vecData ) )
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
            m_pRoot = new CBPNode< uMaximumDepth, uDescriptorSizeBits >( p_vecDescriptorsNEW );
        }

#endif

        //ds tree stats
        uint64_t uDepth            = 0;
        uint64_t uNumberOfEndNodes = 0;
        _setInfoRecursive( m_pRoot, uDepth, uNumberOfEndNodes );
        m_uTotalNumberOfDescriptors += ( p_vecDescriptorsNEW.size( )-uNumberOfNonAggregations );

#if defined REBUILD_BPITREE

        assert( m_uTotalNumberOfDescriptors == m_vecTotalDescriptors.size( ) );
        std::ofstream ofLogFile( "logs/growth_rbpitree.txt", std::ofstream::out | std::ofstream::app );

#else

        std::ofstream ofLogFile( "logs/growth_bpitree.txt", std::ofstream::out | std::ofstream::app );

#endif

        //ds log aggregation
        ofLogFile << p_vecDescriptorsNEW.front( ).uIDKeyFrame
                  << " " << p_vecDescriptorsNEW.size( )
                  << " " << uNumberOfNonAggregations
                  << " " << uDepth
                  << " " << static_cast< double >( m_uTotalNumberOfDescriptors )/uNumberOfEndNodes
                  << " " << m_uTotalNumberOfDescriptors
                  << " " << uNumberOfEndNodes << "\n";
        ofLogFile.close( );

    }

    void match( const std::vector< CPDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptorsQUERY, const UIDKeyFrame& p_uIDQUERY, std::vector< cv::DMatch >& p_vecMatches ) const
    {
        //ds matched training IDs - we only match one point to another - for each key frame
        std::vector< std::set< uint64_t > > vecMatchedIDsTRAIN( p_uIDQUERY );

        //ds for each descriptor
        for( const CPDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorQUERY: p_vecDescriptorsQUERY )
        {
            //ds traverse tree to find this descriptor
            const CBPNode< uMaximumDepth, uDescriptorSizeBits >* pNodeCurrent = m_pRoot;
            while( pNodeCurrent )
            {
                //ds if this node has leaves (is splittable)
                if( pNodeCurrent->bHasLeaves )
                {
                    //ds check the split bit and go deeper
                    if( 0.5 < cDescriptorQUERY.vecData[pNodeCurrent->uIndexSplitBit] )
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
                    for( const CPDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorTRAIN: pNodeCurrent->vecDescriptors )
                    {
                        assert( cDescriptorTRAIN.uIDKeyFrame < p_uIDQUERY );

                        //ds if not matched yet
                        if( 0 == vecMatchedIDsTRAIN[cDescriptorTRAIN.uIDKeyFrame].count( cDescriptorTRAIN.uID ) )
                        {
                            if( uMaximumDistanceHammingProbability > CBPNode< uMaximumDepth, uDescriptorSizeBits >::getDistanceHammingProbability( cDescriptorQUERY.vecData, cDescriptorTRAIN.vecData ) )
                            {
                                //ds add all matches for different key frames - NOT BREAKING
                                p_vecMatches.push_back( cv::DMatch( cDescriptorQUERY.uID, cDescriptorTRAIN.uID, cDescriptorTRAIN.uIDKeyFrame, uMaximumDistanceHammingProbability ) );
                                vecMatchedIDsTRAIN[cDescriptorTRAIN.uIDKeyFrame].insert( cDescriptorTRAIN.uID );
                            }
                        }
                    }
                    break;
                }
            }
        }
    }

    //ds direct matching function on this tree
    void match( const std::vector< CPDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptorsQUERY, std::vector< cv::DMatch >& p_vecMatches ) const
    {
        //ds matched training IDs - we only match one point to another
        std::set< uint64_t > setMatchedIDsTRAIN;

        //ds for each descriptor
        for( const CPDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorQUERY: p_vecDescriptorsQUERY )
        {
            //ds traverse tree to find this descriptor
            const CBPNode< uMaximumDepth, uDescriptorSizeBits >* pNodeCurrent = m_pRoot;
            while( pNodeCurrent )
            {
                //ds if this node has leaves (is splittable)
                if( pNodeCurrent->bHasLeaves )
                {
                    //ds check the split bit and go deeper
                    if( 0.5 < cDescriptorQUERY.vecData[pNodeCurrent->uIndexSplitBit] )
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
                    for( const CPDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorTRAIN: pNodeCurrent->vecDescriptors )
                    {
                        //ds if not matched already
                        if( 0 == setMatchedIDsTRAIN.count( cDescriptorTRAIN.uID ) )
                        {
                            //ds if distance is acceptable
                            if( uMaximumDistanceHammingProbability > CBPNode< uMaximumDepth, uDescriptorSizeBits >::getDistanceHammingProbability( cDescriptorQUERY.vecData, cDescriptorTRAIN.vecData ) )
                            {
                                //ds register match
                                setMatchedIDsTRAIN.insert( cDescriptorTRAIN.uID );

                                //ds add to data structure and exit
                                p_vecMatches.push_back( cv::DMatch( cDescriptorQUERY.uID, cDescriptorTRAIN.uID, uMaximumDistanceHammingProbability ) );
                                break;
                            }
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
            std::vector< const CBPNode< uMaximumDepth, uDescriptorSizeBits >* > vecNodes;

            //ds set vector
            _setNodesRecursive( m_pRoot, vecNodes );

            //ds free nodes
            for( const CBPNode< uMaximumDepth, uDescriptorSizeBits >* pNode: vecNodes )
            {
                delete pNode;
            }

            //ds free all nodes
            vecNodes.clear( );
        }
    }

    //ds constant root node access for reading
    const CBPNode< uMaximumDepth, uDescriptorSizeBits >* getRoot( ) const { return m_pRoot; }

//ds helpers
private:

    void _setNodesRecursive( const CBPNode< uMaximumDepth, uDescriptorSizeBits >* p_pNode, std::vector< const CBPNode< uMaximumDepth, uDescriptorSizeBits >* >& p_vecNodes ) const
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

    void _setInfoRecursive( const CBPNode< uMaximumDepth, uDescriptorSizeBits >* p_pNode, uint64_t& p_uMaximumDepth, uint64_t& p_uNumberOfEndNodes ) const
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

#endif //CBPITREE_H
