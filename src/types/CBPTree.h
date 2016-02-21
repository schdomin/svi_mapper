#ifndef CBPTREE_H
#define CBPTREE_H

#include "CBPNode.h"



template< uint32_t uMaximumDistanceHammingProbability = 50, uint64_t uMaximumDepth = 50, uint32_t uDescriptorSizeBits = 256 >
class CBPTree
{

//ds ctor/dtor
public:

    //ds construct tree upon allocation on filtered descriptors
    CBPTree( const uint64_t& p_uID, const std::vector< CPDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors ): uID( p_uID ),
                                                                                                                       m_pRoot( new CBPNode< uMaximumDepth, uDescriptorSizeBits >( p_vecDescriptors ) )
    {
        assert( 0 != m_pRoot );
    }

    //ds free all nodes in the tree
    ~CBPTree( )
    {
        //ds erase all nodes
        displant( );
    }

//ds control fields
public:

    const uint64_t uID;

private:

    const CBPNode< uMaximumDepth, uDescriptorSizeBits >* m_pRoot;

//ds access
public:

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

        assert( setMatchedIDsTRAIN.size( ) <= p_vecDescriptorsQUERY.size( ) );
    }

    //ds grow the tree
    void plant( const std::vector< CPDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors )
    {
        //ds grow tree on root
        m_pRoot = new CBPNode< uMaximumDepth, uDescriptorSizeBits >( p_vecDescriptors );
        std::printf( "(CBPTree) planted tree with descriptors: %lu\n", p_vecDescriptors.size( ) );
    }

    //ds delete tree
    void displant( )
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

        vecNodes.clear( );
    }

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

};

#endif //CBPTREE_H
