#ifndef CBTREE_H
#define CBTREE_H



#include "CBRIEFNode.h"



template< uint64_t uMaximumDepth, uint32_t uDescriptorSize >
class CBTree
{

//ds eigen memory alignment
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//ds ctor/dtor
public:

    CBTree( const std::vector< CDescriptorBRIEF >& p_vecDescriptors ): m_pRoot( new CBRIEFNode< uMaximumDepth, uDescriptorSize >( p_vecDescriptors ) )
    {
        assert( 0 != m_pRoot );
        std::printf( "(CBTree) allocated tree with descriptors: %lu (%ubit)\n", p_vecDescriptors.size( ), uDescriptorSize );
    }
    ~CBTree( )
    {
        //ds must not be zero
        assert( 0 != m_pRoot );

        //ds nodes holder
        std::vector< const CBRIEFNode< uMaximumDepth, uDescriptorSize >* > vecNodes;

        //ds set vector
        _setNodesRecursive( m_pRoot, vecNodes );

        //ds free nodes
        for( const CBRIEFNode< uMaximumDepth, uDescriptorSize >* pNode: vecNodes )
        {
            delete pNode;
        }

        //ds must be freed now
        assert( 0 == m_pRoot );

        //ds free all nodes
        std::printf( "(CBTree) deallocated nodes: %lu\n", vecNodes.size( ) );
        vecNodes.clear( );
    }



//ds control fields
private:

    const CBRIEFNode< uMaximumDepth, uDescriptorSize >* m_pRoot;

//ds access
public:

    const std::vector< CDescriptorBRIEF >::size_type getNumberOfMatches( const std::vector< CDescriptorBRIEF >& p_vecDescriptorsQuery, const uint32_t& p_uMaximumDistanceHamming ) const
    {
        //ds match count
        std::vector< CDescriptorBRIEF >::size_type uNumberOfMatches = 0;

        //ds for each descriptor
        for( const CDescriptorBRIEF& cDescriptorQuery: p_vecDescriptorsQuery )
        {
            //ds traverse tree to find this descriptor
            const CBRIEFNode< uMaximumDepth, uDescriptorSize >* pNodeCurrent = m_pRoot;
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
                        if( p_uMaximumDistanceHamming > CWrapperOpenCV::getDistanceHamming( cDescriptorQuery, cDescriptorReference ) )
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

    const std::vector< CDescriptorBRIEF >::size_type getNumberOfMatchesFirst( const std::vector< CDescriptorBRIEF >& p_vecDescriptorsQuery, const uint32_t& p_uMaximumDistanceHamming ) const
    {
        //ds match count
        std::vector< CDescriptorBRIEF >::size_type uNumberOfMatches = 0;

        //ds for each descriptor
        for( const CDescriptorBRIEF& cDescriptorQuery: p_vecDescriptorsQuery )
        {
            //ds traverse tree to find this descriptor
            const CBRIEFNode< uMaximumDepth, uDescriptorSize >* pNodeCurrent = m_pRoot;
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
                    if( p_uMaximumDistanceHamming > CWrapperOpenCV::getDistanceHamming( cDescriptorQuery, pNodeCurrent->vecDescriptors.front( ) ) )
                    {
                        ++uNumberOfMatches;
                    }
                    break;
                }
            }
        }

        return uNumberOfMatches;
    }


//ds helpers
private:

    const void _setNodesRecursive( const CBRIEFNode< uMaximumDepth, uDescriptorSize >* p_pNode, std::vector< const CBRIEFNode< uMaximumDepth, uDescriptorSize >* >& p_vecNodes )
    {
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

#endif //CBTREE_H
