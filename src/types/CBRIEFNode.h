#ifndef CBRIEFNODE_H
#define CBRIEFNODE_H

#include "Typedefs.h"

#define MAXIMUM_DEPTH 3


template< uint64_t uMaximumDepth, uint32_t uDescriptorSize >
class CBRIEFNode
{

//ds ctor/dtor
public:

    //ds access only through this constructor
    CBRIEFNode( const uint64_t& p_uDepth,
                const std::vector< CDescriptorBRIEF >& p_vecDescriptors ): CBRIEFNode( p_uDepth, p_vecDescriptors, CDescriptorBRIEF::Ones( ) )
    {
        //ds wrapped
    }

private:

    //ds only internally called
    CBRIEFNode( const uint64_t& p_uDepth,
                const std::vector< CDescriptorBRIEF >& p_vecDescriptors,
                CDescriptorBRIEF p_cMask ): uDepth( p_uDepth ), vecDescriptors( p_vecDescriptors )
    {
        assert( 0 != p_vecDescriptors.size( ) );

        //ds we have to find the split for this node - scan all index
        for( uint32_t uIndexBit = 0; uIndexBit < uDescriptorSize; ++uIndexBit )
        {
            //ds if this index is available in the mask
            if( p_cMask[uIndexBit] )
            {
                //ds compute distance for this index (0.0 is perfect)
                const float fPartitioningCurrent = std::fabs( 0.5-_getOnesFraction( uIndexBit, p_vecDescriptors, uOnesTotal ) );

                //ds if better
                if( fOnesFraction > fPartitioningCurrent )
                {
                    fOnesFraction  = fPartitioningCurrent;
                    uIndexSplitBit = uIndexBit;

                    //ds finalize loop if maximum target is reached
                    if( 0.0 == fOnesFraction )
                    {
                        break;
                    }
                }
            }
        }

        //ds if best was found - we can spawn leaves
        if( -1 != uIndexSplitBit && uMaximumDepth > p_uDepth )
        {
            //ds enabled
            bHasLeaves = true;

            //ds log split
            //std::printf( "[%03lu] split (%0.1f) for index: %03i/%03u (descriptors: %lu)\n", p_uLevel, fOnesFraction, uIndexSplitBit, p_uDescriptorSize, p_vecDescriptors.size( ) );

            //ds update mask
            p_cMask[uIndexSplitBit] = 0;

            //ds first we have to split the descriptors by the found index - preallocate vectors since we know how many ones we have
            std::vector< CDescriptorBRIEF > vecDescriptorsLeafOnes( 0 );
            vecDescriptorsLeafOnes.reserve( uOnesTotal );
            std::vector< CDescriptorBRIEF > vecDescriptorsLeafZeros( 0 );
            vecDescriptorsLeafZeros.reserve( p_vecDescriptors.size( )-uOnesTotal );

            //ds loop over all descriptors and assing them to the new vectors
            for( const CDescriptorBRIEF& cDescriptor: p_vecDescriptors )
            {
                //ds check if split bit is one
                if( cDescriptor[uIndexSplitBit] )
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
            pLeafOnes = new CBRIEFNode( uDepth+1, vecDescriptorsLeafOnes, p_cMask );

            assert( 0 < vecDescriptorsLeafZeros.size( ) );
            pLeafZeros = new CBRIEFNode( uDepth+1, vecDescriptorsLeafZeros, p_cMask );
        }

        //ds free old mask
        //delete[] p_cMask;

    }

public:

    virtual ~CBRIEFNode( )
    {

    }

//ds fields
public:

    //ds rep
    const uint64_t uDepth;
    const std::vector< CDescriptorBRIEF > vecDescriptors;
    int32_t uIndexSplitBit = -1;
    uint32_t uOnesTotal    = 0;
    bool bHasLeaves        = false;
    float fOnesFraction    = 1.0;

    //ds peer: each node has two potential children
    CBRIEFNode* pLeafOnes  = 0;
    CBRIEFNode* pLeafZeros = 0;

//ds helpers
private:

    //ds helpers
    const float _getOnesFraction( const uint64_t& p_uIndexSplitBit, const std::vector< CDescriptorBRIEF >& p_vecDescriptors, uint32_t& p_uOnesTotal ) const
    {
        assert( 0 != p_vecDescriptors.size( ) );

        //ds count
        uint64_t uNumberOfOneBits = 0;

        //ds just add the bits up (a one counts automatically as one)
        for( const CDescriptorBRIEF& cDescriptor: p_vecDescriptors )
        {
            uNumberOfOneBits += cDescriptor[p_uIndexSplitBit];
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

};

#endif //CBRIEFNODE_H
