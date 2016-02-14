#ifndef CBITREE_H
#define CBITREE_H

#include "CBNode.h"



template< uint32_t uMaximumDistanceHamming = 25, uint64_t uMaximumDepth = 50, uint32_t uDescriptorSizeBits = 256 >
class CBITree
{

//ds ctor/dtor
public:

    //ds construct tree upon allocation
    CBITree( ): m_pRoot( 0 )
    {
        //ds nothing to do
    }

    //ds free all nodes in the tree
    ~CBITree( )
    {
        //ds erase all nodes
        displant( );
    }

private:

    CBNode< uMaximumDepth, uDescriptorSizeBits >* m_pRoot;
    //std::vector< CBNode< uMaximumDepth, uDescriptorSizeBits >* > m_vecEndNodes;

//ds access
public:

    void add( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptorsNEW )
    {
        assert( 0 < p_vecDescriptorsNEW.size( ) );

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
                            //ds add to split
                            pNodeCurrent->vecDescriptorsToSplit.push_back( cDescriptorNEW );

                            //ds split as soon as we got two descriptors
                            if( 1 < pNodeCurrent->vecDescriptorsToSplit.size( ) )
                            {
                                //ds try to spawn leafs on this node
                                if( pNodeCurrent->spawnLeafs( pNodeCurrent->vecDescriptorsToSplit ) )
                                {
                                    //ds success
                                    pNodeCurrent->bHasLeaves = true;
                                    bFailedToInsertDescriptor = false;
                                    pNodeCurrent->vecDescriptorsToSplit.clear( );
                                }
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
        
        //ds log aggregation
        std::ofstream ofLogFile( "logs/aggregation_bitree.txt", std::ofstream::out | std::ofstream::app );
        ofLogFile << p_vecDescriptorsNEW.front( ).uIDKeyFrame << " " << ( 1.0-static_cast< double >( uNumberOfNonAggregations )/p_vecDescriptorsNEW.size( ) ) << " " << uNumberOfNonAggregations << " " << p_vecDescriptorsNEW.size( ) << "\n";
        ofLogFile.close( );

        //std::printf( "(CBITree) non-aggregated portion: %f (%lu/%lu)\n", static_cast< double >( uNumberOfNonAggregations )/p_vecDescriptorsNEW.size( ), uNumberOfNonAggregations, p_vecDescriptorsNEW.size( ) );
    }

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

};

#endif //CBITREE_H
