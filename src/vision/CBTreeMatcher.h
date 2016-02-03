#ifndef CBTREEMATCHER_H
#define CBTREEMATCHER_H

#include <opencv2/core/core.hpp>
#include "../types/CBTree.h"



namespace cv
{

template< uint32_t uMaximumDistanceHamming = 25, uint64_t uMaximumDepth = 50, uint32_t uDescriptorSizeBits = 256 >
class CBTreeMatcher: public DescriptorMatcher
{

//ds ctor/dtor
public:

    CBTreeMatcher( )
    {
        //ds clear forest
        m_vecTreesTrained.clear( );
    }

    virtual ~CBTreeMatcher( )
    {
        //ds free trees
    }

//ds access
public:

    virtual void add( const vector< Mat >& p_vecDescriptors )
    {
        //ds for each set of descriptors
        for( const Mat& matDescriptors: p_vecDescriptors )
        {
            //ds allocate descriptor vector in our format
            std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsEigen;
            vecDescriptorsEigen.reserve( matDescriptors.rows );

            //ds fill the vector
            for( uint64_t u = 0; u < matDescriptors.rows; ++u )
            {
                //ds add the wrapped descriptor
                vecDescriptorsEigen.push_back( CDescriptorBRIEF< uDescriptorSizeBits >( u, CBNode< uMaximumDepth, uDescriptorSizeBits >::getDescriptorEigen( matDescriptors.row( u ) ) ) );
            }

            //ds allocate a new tree for these descriptors
            m_vecTreesTrained.push_back( new CBTree< uMaximumDistanceHamming, uMaximumDepth, uDescriptorSizeBits >( m_vecTreesTrained.size( ), vecDescriptorsEigen ) );
        }

        //ds call superclass (otherwise one might deref 0 with matcher->getTrainDescriptors( )..
        DescriptorMatcher::add( p_vecDescriptors );
    }

    virtual void clear( )
    {
        //ds free trees
        assert( 0 == m_vecTreesTrained.size( ) );
        DescriptorMatcher::clear( );
    }

    virtual bool empty( ) const
    {
        //ds no trees -> nothing to match (never trust the empty function)
        return ( 0 == m_vecTreesTrained.size( ) );
    }

    virtual bool isMaskSupported( ) const
    {
        //ds TODO implement
        return false;
    }

    virtual void train( )
    {
        //ds nothing to do (YET)
    }

    virtual Ptr< DescriptorMatcher > clone( bool p_bEmptyTrainData = false ) const
    {
        //ds return a deep copy of this matcher
        return Ptr< CBTreeMatcher >( );
    }

    //ds not visible from superclass
    const CBTree< uMaximumDistanceHamming, uMaximumDepth, uDescriptorSizeBits >* getTreeByID( const uint64_t& p_uID ) const
    {
        assert( m_vecTreesTrained.size( ) > p_uID );
        assert( p_uID == m_vecTreesTrained[p_uID]->uID );
        return m_vecTreesTrained[p_uID];
    }

//ds matching implementation
protected:

    //ds gets called by super::match with k = 1
    virtual void knnMatchImpl( const Mat& matDescriptorsQUERY, vector< vector< DMatch > >& p_vecMatches, int p_iK, const vector<Mat>& masks=vector<Mat>(), bool compactResult=false )
    {
        //ds get query descriptors to our format - allocate descriptor vector in our format
        vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsQUERY;
        vecDescriptorsQUERY.reserve( matDescriptorsQUERY.rows );

        //ds fill the vector
        for( uint64_t u = 0; u < matDescriptorsQUERY.rows; ++u )
        {
            //ds add the wrapped descriptor
            vecDescriptorsQUERY.push_back( CDescriptorBRIEF< uDescriptorSizeBits >( u, CBNode< uMaximumDepth, uDescriptorSizeBits >::getDescriptorEigen( matDescriptorsQUERY.row( u ) ) ) );
        }

        //ds check if its a regular matching request
        if( 1 == p_iK )
        {
            //ds for each trained tree
            for( const CBTree< uMaximumDistanceHamming, uMaximumDepth, uDescriptorSizeBits >* pTree: m_vecTreesTrained )
            {
                //ds reset point counts
                //pTree->resetPointsPerLeaf( );

                //ds match counting
                //const uint64_t uMatchesBefore = p_vecMatches.size( );

                //ds get matches of that tree
                pTree->setMatches1NN( vecDescriptorsQUERY, pTree->uID, p_vecMatches );

                //ds dump matching
                //pTree->writeStatistics( static_cast< double >( p_vecMatches.size( )-uMatchesBefore )/vecDescriptorsQUERY.size( ), m_vecTreesTrained.size( ) );
            }
        }
        else
        {
            //ds TODO implement
            throw Exception( 0, "radiusMatchImpl not implemented", "radiusMatchImpl", "CBTreeMatcher.h", 0 );
        }
    }

    //ds not implemented
    virtual void radiusMatchImpl( const Mat& queryDescriptors, vector<vector<DMatch> >& matches, float maxDistance, const vector<Mat>& masks=vector<Mat>(), bool compactResult=false )
    {
        //ds escape
        throw Exception( 0, "radiusMatchImpl not implemented", "radiusMatchImpl", "CBTreeMatcher.h", 0 );
    }

//ds control fields
protected:

    //ds trained trees one for each add call
    vector< const CBTree< uMaximumDistanceHamming, uMaximumDepth, uDescriptorSizeBits >* > m_vecTreesTrained;

};

} //namespace cv

#endif //CBTREEMATCHER_H
