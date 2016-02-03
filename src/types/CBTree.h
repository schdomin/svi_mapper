#ifndef CBTREE_H
#define CBTREE_H

#include "CBNode.h"



//TODO templatify KeyFrame for maximum generics
#define MAXIMUM_DISTANCE_HAMMING 25
#define BTREE_MAXIMUM_DEPTH 50
#define DESCRIPTOR_SIZE_BITS 256
#define DESCRIPTOR_SIZE_BYTES DESCRIPTOR_SIZE_BITS/8

#define USING_BTREE
//#define USING_BF
//#define USING_LSH
//#define USING_BOW



template< uint32_t uMaximumDistanceHamming = 25, uint64_t uMaximumDepth = 50, uint32_t uDescriptorSizeBits = 256 >
class CBTree
{

//ds eigen memory alignment
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//ds ctor/dtor
public:

    //ds construct tree upon allocation
    CBTree( const uint64_t& p_uID, const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors ): uID( p_uID ), m_pRoot( new CBNode< uMaximumDepth, uDescriptorSizeBits >( p_vecDescriptors ) )
    {
        //m_vecEndNodes.clear( );
        assert( 0 != m_pRoot );
        //std::printf( "(CBTree) allocated tree [%06lu] with descriptors: %lu\n", p_uID, p_vecDescriptors.size( ) );

        //ds get nodes for information
        //_setEndNodesRecursive( m_pRoot, m_vecEndNodes );
        //std::printf( "(CBTree)                           with end nodes: %lu(%3.1f)\n", m_vecEndNodes.size( ), static_cast< double >( m_vecEndNodes.size( ) )/p_vecDescriptors.size( ) );
    }

    //ds free all nodes in the tree
    ~CBTree( )
    {
        //ds erase all nodes
        displant( );
    }



//ds control fields
public:

    const uint64_t uID;

private:

    const CBNode< uMaximumDepth, uDescriptorSizeBits >* m_pRoot;
    //std::vector< CBNode< uMaximumDepth, uDescriptorSizeBits >* > m_vecEndNodes;

//ds access
public:

    const uint64_t getNumberOfMatches( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptorsQuery, const uint32_t& p_uMaximumDistanceHamming ) const
    {
        //ds match count
        uint64_t uNumberOfMatches = 0;

        //ds for each descriptor
        for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorQuery: p_vecDescriptorsQuery )
        {
            //ds traverse tree to find this descriptor
            const CBNode< uMaximumDepth, uDescriptorSizeBits >* pNodeCurrent = m_pRoot;
            while( pNodeCurrent )
            {
                //ds if this node has leaves (is splittable)
                if( pNodeCurrent->bHasLeaves )
                {
                    //ds check the split bit and go deeper
                    if( cDescriptorQuery.vecData[pNodeCurrent->uIndexSplitBit] )
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
                    for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorReference: pNodeCurrent->vecDescriptors )
                    {
                        if( p_uMaximumDistanceHamming > CBNode< uMaximumDepth, uDescriptorSizeBits >::getDistanceHamming( cDescriptorQuery.vecData, cDescriptorReference.vecData ) )
                        {
                            ++uNumberOfMatches;
                            break;
                        }
                    }
                    break;
                }
            }
        }

        return uNumberOfMatches;
    }

    const uint64_t getNumberOfMatchesLazyEvaluation( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptorsQuery, const uint32_t& p_uMaximumDistanceHamming ) const
    {
        //ds match count
        uint64_t uNumberOfMatches = 0;

        //ds for each descriptor
        for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorQuery: p_vecDescriptorsQuery )
        {
            //ds traverse tree to find this descriptor
            const CBNode< uMaximumDepth, uDescriptorSizeBits >* pNodeCurrent = m_pRoot;
            while( pNodeCurrent )
            {
                //ds if this node has leaves (is splittable)
                if( pNodeCurrent->bHasLeaves )
                {
                    //ds check the split bit and go deeper
                    if( cDescriptorQuery.vecData[pNodeCurrent->uIndexSplitBit] )
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
                    if( p_uMaximumDistanceHamming > CBNode< uMaximumDepth, uDescriptorSizeBits >::getDistanceHamming( cDescriptorQuery.vecData, pNodeCurrent->vecDescriptors.front( ).vecData ) )
                    {
                        ++uNumberOfMatches;
                    }
                    break;
                }
            }
        }

        return uNumberOfMatches;
    }

    void setMatches1NN( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptorsQUERY, const uint32_t& p_uIDTrain, std::vector< std::vector< cv::DMatch > >& p_vecMatches ) const
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
                            p_vecMatches.push_back( std::vector< cv::DMatch >( 1, cv::DMatch( cDescriptorQUERY.uID, cDescriptorTRAIN.uID, p_uIDTrain, uMaximumDistanceHamming ) ) );
                            break;
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

    //ds grow the tree
    void plant( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors )
    {
        //ds grow tree on root
        m_pRoot = new CBNode< uMaximumDepth, uDescriptorSizeBits >( p_vecDescriptors );
        std::printf( "(CBTree) planted tree with descriptors: %lu\n", p_vecDescriptors.size( ) );
    }

    //ds delete tree
    void displant( )
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
        //std::printf( "(CBTree) deallocated nodes: %lu\n", vecNodes.size( ) );
        vecNodes.clear( );
    }

    /*ds info (functions do nothing if structures are not set)
    void printPointsPerLeaf( ) const
    {
        //ds list all nodes with points
        for( typename std::vector< CBNode< uMaximumDepth, uDescriptorSizeBits >* >::size_type u = 0; u < m_vecEndNodes.size( ); ++u )
        {
            if( 0 < m_vecEndNodes[u]->uLinkedPoints )
            {
                std::printf( "[%06lu][%06lu][%02lu] points: %lu\n", uID, u, m_vecEndNodes[u]->uDepth, m_vecEndNodes[u]->uLinkedPoints );
            }
        }
    }

    void resetPointsPerLeaf( )
    {
        //ds for all nodes
        for( CBNode< uMaximumDepth, uDescriptorSizeBits >* pNode: m_vecEndNodes )
        {
            pNode->uLinkedPoints = 0;
        }
    }

    void writeStatistics( const double& p_dRelativeMatches, const uint64_t& p_uIDTreeQUERY ) const
    {
        //ds construct filename for points per leaf
        char chBufferPPL[256];
        std::snprintf( chBufferPPL, 256, "logs/tree_%06lu_ppl.txt", uID );

        //ds open file
        std::ofstream ofLogfilePPL( chBufferPPL, std::ofstream::out | std::ofstream::app );

        //ds total points matched count
        uint64_t uLinkedPointsTotal = 0;

        //ds write info (one line)
        for( typename std::vector< CBNode< uMaximumDepth, uDescriptorSizeBits >* >::size_type u = 0; u < m_vecEndNodes.size( ); ++u )
        {
            uLinkedPointsTotal += m_vecEndNodes[u]->uLinkedPoints;
            ofLogfilePPL << m_vecEndNodes[u]->uLinkedPoints << " ";
        }
        ofLogfilePPL << "\n";
        ofLogfilePPL.close( );

        //ds construct filename for closing stats
        char chBufferMatching[256];
        std::snprintf( chBufferMatching, 256, "logs/tree_%06lu_matching.txt", uID );

        //ds open file write and save
        std::ofstream ofLogfileMatching( chBufferMatching, std::ofstream::out | std::ofstream::app );
        ofLogfileMatching << uID << " " << p_uIDTreeQUERY << " " << uLinkedPointsTotal << " " << p_dRelativeMatches << "\n";
        ofLogfileMatching.close( );
    }*/

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

#endif //CBTREE_H
