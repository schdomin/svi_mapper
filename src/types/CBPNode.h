#ifndef CBPNODE_H
#define CBPNODE_H

#include <vector>
#include <bitset>

#include "CPDescriptorBRIEF.h"



template< uint64_t uMaximumDepth = 50, uint32_t uDescriptorSizeBits = 256 >
class CBPNode
{

    //ds readability
    using CDescriptorVector = Eigen::Matrix< double, uDescriptorSizeBits, 1 >;

//ds ctor/dtor
public:

    //ds access only through this constructor
    CBPNode( const std::vector< CPDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors ): CBPNode< uMaximumDepth, uDescriptorSizeBits >( 0, p_vecDescriptors, _getMaskClean( ) )
    {
        //ds wrapped
    }

    //ds create leafs (external use intended)
    bool spawnLeafs( )
    {
        //ds if there are at least 2 descriptors (minimal split)
        if( 1 < vecDescriptors.size( ) )
        {
            assert( !bHasLeaves );

            //ds affirm initial situation
            uIndexSplitBit = -1;
            uOnesTotal     = 0;
            dPartitioning  = 1.0;

            //ds we have to find the split for this node - scan all index
            for( uint32_t uIndexBit = 0; uIndexBit < uDescriptorSizeBits; ++uIndexBit )
            {
                //ds if this index is available in the mask
                if( vecMask[uIndexBit] )
                {
                    //ds temporary set bit count
                    uint32_t uNumberOfSetBits = 0;

                    //ds compute distance for this index (0.0 is perfect)
                    const double fPartitioningCurrent = std::fabs( 0.5-_getOnesFraction( uIndexBit, vecDescriptors, uNumberOfSetBits ) );

                    //ds if better
                    if( dPartitioning > fPartitioningCurrent )
                    {
                        dPartitioning  = fPartitioningCurrent;
                        uOnesTotal     = uNumberOfSetBits;
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
            if( -1 != uIndexSplitBit && uMaximumDepth > uDepth )
            {
                //ds check if we have enough data to split (NOT REQUIRED IF DEPTH IS SET ACCORDINGLY)
                if( 0 < uOnesTotal && 0.5 > dPartitioning )
                {
                    //ds enabled
                    bHasLeaves = true;

                    //ds get a mask copy
                    std::bitset< uDescriptorSizeBits > vecMask( vecMask );

                    //ds update mask for leafs
                    vecMask[uIndexSplitBit] = 0;

                    //ds first we have to split the descriptors by the found index - preallocate vectors since we know how many ones we have
                    std::vector< CPDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsLeafOnes;
                    vecDescriptorsLeafOnes.reserve( uOnesTotal );
                    std::vector< CPDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsLeafZeros;
                    vecDescriptorsLeafZeros.reserve( vecDescriptors.size( )-uOnesTotal );

                    //ds loop over all descriptors and assing them to the new vectors
                    for( const CPDescriptorBRIEF< uDescriptorSizeBits >& cDescriptor: vecDescriptors )
                    {
                        //ds check if split bit is set
                        if( 0.5 < cDescriptor.vecData[uIndexSplitBit] )
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
                    pLeafOnes = new CBPNode< uMaximumDepth, uDescriptorSizeBits >( uDepth+1, vecDescriptorsLeafOnes, vecMask );

                    assert( 0 < vecDescriptorsLeafZeros.size( ) );
                    pLeafZeros = new CBPNode< uMaximumDepth, uDescriptorSizeBits >( uDepth+1, vecDescriptorsLeafZeros, vecMask );

                    //ds worked
                    return true;
                }
                else
                {
                    //ds split failed
                    return false;
                }
            }
            else
            {
                //ds split failed
                return false;
            }
        }
        else
        {
            return false;
        }
    }

private:

    //ds only internally called
    CBPNode( const uint64_t& p_uDepth,
            const std::vector< CPDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors,
            std::bitset< uDescriptorSizeBits > p_vecMask ): uDepth( p_uDepth ), vecDescriptors( p_vecDescriptors ), vecMask( p_vecMask )
    {
        //ds call recursive leaf spawner
        spawnLeafs( );
    }

public:

    CBPNode( )
    {
        //ds nothing to do (the leafs will be freed manually)
    }

//ds fields
public:

    //ds rep
    const uint64_t uDepth;
    std::vector< CPDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptors;
    int32_t uIndexSplitBit = -1;
    uint32_t uOnesTotal    = 0;
    bool bHasLeaves        = false;
    double dPartitioning   = 1.0;
    std::bitset< uDescriptorSizeBits > vecMask;

    //ds info (incremented in tree during search)
    //uint64_t uLinkedPoints = 0;

    //ds peer: each node has two potential children
    CBPNode* pLeafOnes  = 0;
    CBPNode* pLeafZeros = 0;

//ds helpers
private:

    //ds helpers
    const double _getOnesFraction( const uint64_t& p_uIndexSplitBit, const std::vector< CPDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors, uint32_t& p_uOnesTotal ) const
    {
        assert( 0 < p_vecDescriptors.size( ) );

        //ds count
        uint64_t uNumberOfOneBits = 0;

        //ds just add the bits up (a one counts automatically as one)
        for( const CPDescriptorBRIEF< uDescriptorSizeBits >& cDescriptor: p_vecDescriptors )
        {
            //ds if the probability for one is more than average
            if( 0.5 < cDescriptor.vecData[p_uIndexSplitBit] )
            {
                ++uNumberOfOneBits;
            }
        }

        //ds set total
        p_uOnesTotal = uNumberOfOneBits;
        assert( p_uOnesTotal <= p_vecDescriptors.size( ) );

        //ds return ratio
        return ( static_cast< float >( uNumberOfOneBits )/p_vecDescriptors.size( ) );
    }

    //ds returns a bitset with all bits set to 1.0
    std::bitset< uDescriptorSizeBits > _getMaskClean( ) const
    {
        std::bitset< uDescriptorSizeBits > vecMask;
        vecMask.set( );
        return vecMask;
    }

//ds format
public:

    //ds converts descriptors from cv::Mat to the current descriptor vector format
    inline static const CDescriptorVector getDescriptorVector( const cv::Mat& p_cDescriptor )
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
    inline static const uint32_t getDistanceHammingProbability( const CDescriptorVector& p_vecDescriptorQuery,
                                                                const CDescriptorVector& p_vecDescriptorReference )
    {
        //ds score
        double dProbableHammingDistance = 0.0;

        //ds for all elements
        for( uint32_t u = 0; u < uDescriptorSizeBits; ++u )
        {
            //ds D(p1, p2)= sum_i dist(p1[i],p2[i]) -> dist(p1[i],p2[2]) is p(p1[i]=1)*p(p2[i]=0)+p(p1[i]=0)*p(p2[i]=1)=p(p1[i]=1)*(1-p(p2[i]=1))+(1-p(p1[i]=1))*p(p2[i]=1)
            dProbableHammingDistance += p_vecDescriptorQuery[u] + p_vecDescriptorReference[u] - 2*p_vecDescriptorQuery[u]*p_vecDescriptorReference[u];
        }

        //ds return the distance
        return dProbableHammingDistance;
    }

};

#endif //CBPNODE_H
