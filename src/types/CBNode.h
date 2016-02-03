#ifndef CBNODE_H
#define CBNODE_H

#include <vector>
#include "CDescriptorBRIEF.h"



template< uint64_t uMaximumDepth = 50, uint32_t uDescriptorSizeBits = 256 >
class CBNode
{

    //ds readability
    using CDescriptorVector = Eigen::Matrix< bool, uDescriptorSizeBits, 1 >;

//ds eigen memory alignment
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

//ds ctor/dtor
public:

    //ds access only through this constructor
    CBNode( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors ): CBNode< uMaximumDepth, uDescriptorSizeBits >( 0, p_vecDescriptors, CDescriptorVector::Ones( ) )
    {
        //ds wrapped
    }

private:

    //ds only internally called
    CBNode( const uint64_t& p_uDepth,
            const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors,
            CDescriptorVector p_cMask ): uDepth( p_uDepth ), vecDescriptors( p_vecDescriptors )
    {
        assert( 0 != p_vecDescriptors.size( ) );

        //ds we have to find the split for this node - scan all index
        for( uint32_t uIndexBit = 0; uIndexBit < uDescriptorSizeBits; ++uIndexBit )
        {
            //ds if this index is available in the mask
            if( p_cMask[uIndexBit] )
            {
                //ds compute distance for this index (0.0 is perfect)
                const double fPartitioningCurrent = std::fabs( 0.5-_getOnesFraction( uIndexBit, p_vecDescriptors, uOnesTotal ) );

                //ds if better
                if( dPartitioning > fPartitioningCurrent )
                {
                    dPartitioning  = fPartitioningCurrent;
                    uIndexSplitBit = uIndexBit;

                    //ds finalize loop if maximum target is reached
                    if( 0.0 == dPartitioning )
                    {
                        break;
                    }
                }
            }
        }

        //ds if best was found - we can spawn leaves
        if( -1 != uIndexSplitBit && uMaximumDepth > p_uDepth )
        {
            //ds check if we have enough data to split (NOT REQUIRED IF DEPTH IS SET ACCORDINGLY)
            if( 0 < uOnesTotal && 0.5 > dPartitioning )
            {
                //ds enabled
                bHasLeaves = true;

                //ds update mask
                p_cMask[uIndexSplitBit] = 0;

                //ds first we have to split the descriptors by the found index - preallocate vectors since we know how many ones we have
                std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsLeafOnes;
                vecDescriptorsLeafOnes.reserve( uOnesTotal );
                std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsLeafZeros;
                vecDescriptorsLeafZeros.reserve( p_vecDescriptors.size( )-uOnesTotal );

                //ds loop over all descriptors and assing them to the new vectors
                for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptor: p_vecDescriptors )
                {
                    //ds check if split bit is one
                    if( cDescriptor.vecData[uIndexSplitBit] )
                    {
                        vecDescriptorsLeafOnes.push_back( cDescriptor );
                    }
                    else
                    {
                        vecDescriptorsLeafZeros.push_back( cDescriptor );
                    }
                }

                //ds if there are elements for leaves
                assert( 0 < vecDescriptorsLeafOnes.size( ) );
                pLeafOnes = new CBNode< uMaximumDepth, uDescriptorSizeBits >( uDepth+1, vecDescriptorsLeafOnes, p_cMask );

                assert( 0 < vecDescriptorsLeafZeros.size( ) );
                pLeafZeros = new CBNode< uMaximumDepth, uDescriptorSizeBits >( uDepth+1, vecDescriptorsLeafZeros, p_cMask );
            }
            else
            {
                //ds if we got a final node with more than one descriptor
                if( 1 < p_vecDescriptors.size( ) )
                {
                    //ds compute internal difference
                    uint32_t uDiversity = 0;
                    const CDescriptorBRIEF< uDescriptorSizeBits > cDescriptorReference( p_vecDescriptors.front( ) );
                    for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorInner: p_vecDescriptors )
                    {
                        uDiversity += CBNode< uMaximumDepth, uDescriptorSizeBits >::getDistanceHamming( cDescriptorInner.vecData, cDescriptorReference.vecData );
                    }

                    //ds if theres no differences we can reduce the vector to one single element
                    if( 0 == uDiversity )
                    {
                        //ds reduce vector to one element
                        vecDescriptors.clear( );
                        vecDescriptors.push_back( cDescriptorReference );
                    }
                }
            }
        }
    }

public:

    CBNode( )
    {
        //ds nothing to do (the leafs will be freed manually)
    }

//ds fields
public:

    //ds rep
    const uint64_t uDepth;
    std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptors;
    int32_t uIndexSplitBit = -1;
    uint32_t uOnesTotal    = 0;
    bool bHasLeaves        = false;
    double dPartitioning   = 1.0;

    //ds info (incremented in tree during search)
    //uint64_t uLinkedPoints = 0;

    //ds peer: each node has two potential children
    const CBNode* pLeafOnes  = 0;
    const CBNode* pLeafZeros = 0;

//ds helpers
private:

    //ds helpers
    const double _getOnesFraction( const uint64_t& p_uIndexSplitBit, const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors, uint32_t& p_uOnesTotal ) const
    {
        assert( 0 != p_vecDescriptors.size( ) );

        //ds count
        uint64_t uNumberOfOneBits = 0;

        //ds just add the bits up (a one counts automatically as one)
        for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptor: p_vecDescriptors )
        {
            uNumberOfOneBits += cDescriptor.vecData[p_uIndexSplitBit];
        }

        //ds set total
        p_uOnesTotal = uNumberOfOneBits;

        //ds return ratio
        return ( static_cast< float >( uNumberOfOneBits )/p_vecDescriptors.size( ) );
    }

    /*CDescriptor _getCopy( const CDescriptor& p_cMask ) const
    {
        CDescriptor cMaskCopy = new CDescriptorBRIEFElement[uDescriptorSize];
        for( uint32_t u = 0; u < uDescriptorSize; ++u )
        {
            cMaskCopy[u] = p_cMask[u];
        }
        return cMaskCopy;
    }*/

    /*CDescriptorBRIEF _getMaskClean( ) const
    {
        CDescriptorBRIEF cMask( 1, uDescriptorSize, CV_8U ); // = new CDescriptorBRIEFElement[uDescriptorSize];
        for( uint32_t u = 0; u < uDescriptorSize; ++u )
        {
            cMask[u] = 1;
        }
        return cMask;
    }*/

//ds format
public:

    //ds converts descriptors from cv::Mat to Eigen::Matrix
    inline static const CDescriptorVector getDescriptorEigen( const cv::Mat& p_cDescriptor )
    {
        //ds return vector
        CDescriptorVector vecDescriptor;

        //ds compute bytes (as  opencv descriptors are bytewise)
        const uint32_t uDescriptorSizeBytes = uDescriptorSizeBits/8;

        //ds loop over all bytes
        for( uint32_t u = 0; u < uDescriptorSizeBytes; ++u )
        {
            //ds get minimal datafrom cv::mat
            const uchar chValue = p_cDescriptor.at< uchar >( u );

            //ds get bitstring
            for( uint8_t v = 0; v < 8; ++v )
            {
                vecDescriptor[u*8+v] = ( chValue >> v ) & 1;
            }
        }

        return vecDescriptor;
    }

    //ds computes Hamming distance for Eigen::Matrix descriptors
    inline static const uint32_t getDistanceHamming( const CDescriptorVector& p_vecDescriptorQuery,
                                                     const CDescriptorVector& p_vecDescriptorReference )
    {
        //ds count set bits
        return ( p_vecDescriptorQuery-p_vecDescriptorReference ).count( );
    }

};

#endif //CBNODE_H
