#ifndef CBNODE_H
#define CBNODE_H

#include <vector>
#include "CDescriptorBRIEF.h"
#include "Types.h"

#define SPLIT_BALANCED



template< uint64_t uMaximumDepth = 50, uint32_t uDescriptorSizeBits = 256 >
class CBNode
{

    //ds readability
    using CDescriptorVector = std::bitset< uDescriptorSizeBits >; //Eigen::Matrix< bool, uDescriptorSizeBits, 1 >;

//ds ctor/dtor
public:

    //ds access only through this constructor: no mask provided
    CBNode( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors ): CBNode< uMaximumDepth, uDescriptorSizeBits >( 0, p_vecDescriptors, _getMaskClean( ) )
    {
        //ds wrapped
    }

    //ds access only through this constructor: mask provided
    CBNode( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors, CDescriptorVector p_vecBitMask ): CBNode< uMaximumDepth, uDescriptorSizeBits >( 0, p_vecDescriptors, p_vecBitMask )
    {
        //ds wrapped
    }

    //ds access only through this constructor: split order provided
    CBNode( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors,
            std::vector< uint32_t > p_vecSplitOrder ): CBNode< uMaximumDepth, uDescriptorSizeBits >( 0, p_vecDescriptors, p_vecSplitOrder )
    {
        //ds wrapped
    }

    //ds bit statistics
    CBNode( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors,
            const std::map< UIDLandmark, CBitStatistics >& p_mapBitStatistics ): CBNode< uMaximumDepth, uDescriptorSizeBits >( 0, p_vecDescriptors, _getMaskClean( ), p_mapBitStatistics )
    {

    }

    //ds create leafs (external use intented)
    bool spawnLeafs( )
    {
        //ds filter descriptors before leafing
        //_filterDescriptorsExhaustive( );

        //ds if there are at least 2 descriptors (minimal split)
        if( 1 < vecDescriptors.size( ) )
        {
            assert( !bHasLeaves );

            //ds affirm initial situation
            uIndexSplitBit = -1;
            uOnesTotal     = 0;
            dPartitioning  = 1.0;

#ifdef SPLIT_BALANCED

            //ds we have to find the split for this node - scan all index
            for( uint32_t uIndexBit = 0; uIndexBit < uDescriptorSizeBits; ++uIndexBit )
            {
                //ds if this index is available in the mask
                if( matMask[uIndexBit] )
                {
                    //ds temporary set bit count
                    uint64_t uNumberOfSetBits = 0;

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

#else

            //ds we have to find the split for this node - scan all index
            for( uint32_t uIndexBit = 0; uIndexBit < uDescriptorSizeBits; ++uIndexBit )
            {
                //ds if this index is available in the mask
                if( matMask[uIndexBit] )
                {
                    //ds temporary set bit count
                    uint32_t uNumberOfSetBits = 0;

                    //ds compute fraction
                    const double dOnesFraction = _getOnesFraction( uIndexBit, vecDescriptors, uNumberOfSetBits );
                    assert( 0.0 <= dOnesFraction );
                    assert( 1.0 >= dOnesFraction );

                    //ds if we have at least a minimal split
                    if( 0.0 < dOnesFraction && 1.0 > dOnesFraction )
                    {
                        //ds compute distance for this index - we want to have a minimal or maximal ones fraction
                        const double fPartitioningCurrent = std::min( dOnesFraction, 1.0-dOnesFraction );

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
            }

#endif

            //ds if best was found - we can spawn leaves
            if( -1 != uIndexSplitBit && uMaximumDepth > uDepth )
            {
                //ds check if we have enough data to split (NOT REQUIRED IF DEPTH IS SET ACCORDINGLY)
                if( 0 < uOnesTotal && 0.5 > dPartitioning )
                {
                    /*if( 5 > uDepth)
                    {
                        std::cerr << "depth: " << uDepth << " bit: " << uIndexSplitBit << std::endl;
                    }*/

                    //ds enabled
                    bHasLeaves = true;

                    //ds get a mask copy
                    CDescriptorVector vecMask( matMask );

                    //ds update mask for leafs
                    vecMask[uIndexSplitBit] = 0;

                    //ds first we have to split the descriptors by the found index - preallocate vectors since we know how many ones we have
                    std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsLeafOnes;
                    vecDescriptorsLeafOnes.reserve( uOnesTotal );
                    std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsLeafZeros;
                    vecDescriptorsLeafZeros.reserve( vecDescriptors.size( )-uOnesTotal );

                    //ds loop over all descriptors and assing them to the new vectors
                    for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptor: vecDescriptors )
                    {
                        //ds check if split bit is set
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
                    pLeafOnes = new CBNode< uMaximumDepth, uDescriptorSizeBits >( uDepth+1, vecDescriptorsLeafOnes, vecMask );

                    assert( 0 < vecDescriptorsLeafZeros.size( ) );
                    pLeafZeros = new CBNode< uMaximumDepth, uDescriptorSizeBits >( uDepth+1, vecDescriptorsLeafZeros, vecMask );

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

    //ds create leafs following the set split order
    bool spawnLeafs( std::vector< uint32_t > p_vecSplitOrder )
    {
        //ds if there are at least 2 descriptors (minimal split)
        if( 1 < vecDescriptors.size( ) )
        {
            assert( !bHasLeaves );

            //ds affirm initial situation
            uIndexSplitBit = -1;
            uOnesTotal     = 0;
            dPartitioning  = 1.0;

            //uint32_t uShift = 0;
            //uint32_t uBitOptimal = p_vecSplitOrder[uDepth];

            //ds try a selection of available bit splits
            for( uint32_t uDepthTrial = uDepth; uDepthTrial < 2*uDepth+1; ++uDepthTrial )
            {
                uint64_t uOnesTotalCurrent = 0;

                //ds compute distance for this index (0.0 is perfect)
                const double dPartitioningCurrent = std::fabs( 0.5-_getOnesFraction( p_vecSplitOrder[uDepthTrial], vecDescriptors, uOnesTotalCurrent ) );

                if( dPartitioning > dPartitioningCurrent )
                {
                    //ds buffer found bit
                    const uint32_t uSplitBitBest = p_vecSplitOrder[uDepthTrial];

                    //ds shift the last best index to the chosen depth in a later step
                    p_vecSplitOrder[uDepthTrial] = uIndexSplitBit;

                    dPartitioning  = dPartitioningCurrent;
                    uOnesTotal     = uOnesTotalCurrent;
                    uIndexSplitBit = uSplitBitBest;

                    //ds update the split order vector to the current bit
                    p_vecSplitOrder[uDepth] = uSplitBitBest;

                    //uShift = uDepthTrial-uDepth;

                    //ds finalize loop if maximum target is reached
                    if( 0.0 == dPartitioning )
                    {
                        break;
                    }
                }
            }

            //ds if best was found - we can spawn leaves
            if( -1 != uIndexSplitBit && uMaximumDepth > uDepth )
            {
                //ds check if we have enough data to split (NOT REQUIRED IF DEPTH IS SET ACCORDINGLY)
                if( 0 < uOnesTotal && 0.5 > dPartitioning )
                {
                    /*if( 5 > uDepth)
                    {
                        std::cerr << "depth: " << uDepth << " bit: " << uIndexSplitBit << " optimal: " << uBitOptimal <<  " shift: " << uShift << std::endl;
                    }*/

                    //ds enabled
                    bHasLeaves = true;

                    //ds first we have to split the descriptors by the found index - preallocate vectors since we know how many ones we have
                    std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsLeafOnes;
                    vecDescriptorsLeafOnes.reserve( uOnesTotal );
                    std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsLeafZeros;
                    vecDescriptorsLeafZeros.reserve( vecDescriptors.size( )-uOnesTotal );

                    //ds loop over all descriptors and assing them to the new vectors
                    for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptor: vecDescriptors )
                    {
                        //ds check if split bit is set
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
                    pLeafOnes = new CBNode< uMaximumDepth, uDescriptorSizeBits >( uDepth+1, vecDescriptorsLeafOnes, p_vecSplitOrder );

                    assert( 0 < vecDescriptorsLeafZeros.size( ) );
                    pLeafZeros = new CBNode< uMaximumDepth, uDescriptorSizeBits >( uDepth+1, vecDescriptorsLeafZeros, p_vecSplitOrder );

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

    bool spawnLeafs( const std::map< UIDLandmark, CBitStatistics >& p_mapBitStatistics )
    {
        //ds buffer number of descriptors
        const uint64_t uNumberOfDescriptors = vecDescriptors.size( );

        //ds if there are at least 2 descriptors (minimal split)
        if( 1 < uNumberOfDescriptors )
        {
            assert( !bHasLeaves );

            //ds affirm initial situation
            uIndexSplitBit = -1;
            uOnesTotal     = 0;
            dPartitioning  = 1.0;
            double dVarianceMaximum = 0.0;

            //ds variance computation statistics
            Eigen::Matrix< double, uDescriptorSizeBits, 1 > vecAccumulatedProbabilities( Eigen::Matrix< double, uDescriptorSizeBits, 1 >::Zero( ) );

            //ds for all descriptors in this node
            for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptor: vecDescriptors )
            {
                vecAccumulatedProbabilities += p_mapBitStatistics.at( cDescriptor.uIDLandmark ).vecBitProbabilities;
            }

            //ds get average
            const Eigen::Matrix< double, uDescriptorSizeBits, 1 > vecMean( vecAccumulatedProbabilities/uNumberOfDescriptors );
            std::vector< std::pair< uint32_t, double > > vecVariance( uDescriptorSizeBits );

            //ds compute variance
            for( uint32_t uIndexBit = 0; uIndexBit < uDescriptorSizeBits; ++uIndexBit )
            {
                //ds if this index is available in the mask
                if( matMask[uIndexBit] )
                {
                    //ds buffers
                    double dVariance = 0.0;

                    //ds for all descriptors in this node
                    for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptor: vecDescriptors )
                    {
                        const double dDelta = p_mapBitStatistics.at( cDescriptor.uIDLandmark ).vecBitProbabilities[uIndexBit]-vecMean[uIndexBit];
                        dVariance += dDelta*dDelta;
                    }

                    //ds average
                    dVariance /= uNumberOfDescriptors;

                    //ds check if better
                    if( dVarianceMaximum < dVariance )
                    {
                        dVarianceMaximum = dVariance;
                        uIndexSplitBit   = uIndexBit;
                    }
                }
            }

            //ds if best was found - we can spawn leaves
            if( -1 != uIndexSplitBit && uMaximumDepth > uDepth )
            {
                //ds compute distance for this index (0.0 is perfect)
                dPartitioning = std::fabs( 0.5-_getOnesFraction( uIndexSplitBit, vecDescriptors, uOnesTotal ) );

                //ds check if we have enough data to split (NOT REQUIRED IF DEPTH IS SET ACCORDINGLY)
                if( 0 < uOnesTotal && 0.5 > dPartitioning )
                {
                    //ds enabled
                    bHasLeaves = true;

                    //ds get a mask copy
                    CDescriptorVector vecMask( matMask );

                    //ds update mask for leafs
                    vecMask[uIndexSplitBit] = 0;

                    //ds first we have to split the descriptors by the found index - preallocate vectors since we know how many ones we have
                    std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsLeafOnes;
                    vecDescriptorsLeafOnes.reserve( uOnesTotal );
                    std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsLeafZeros;
                    vecDescriptorsLeafZeros.reserve( uNumberOfDescriptors-uOnesTotal );

                    //ds loop over all descriptors and assing them to the new vectors
                    for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptor: vecDescriptors )
                    {
                        //ds check if split bit is set
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
                    pLeafOnes = new CBNode< uMaximumDepth, uDescriptorSizeBits >( uDepth+1, vecDescriptorsLeafOnes, vecMask, p_mapBitStatistics );

                    assert( 0 < vecDescriptorsLeafZeros.size( ) );
                    pLeafZeros = new CBNode< uMaximumDepth, uDescriptorSizeBits >( uDepth+1, vecDescriptorsLeafZeros, vecMask, p_mapBitStatistics );

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
    CBNode( const uint64_t& p_uDepth,
            const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors,
            CDescriptorVector p_vecMask ): uDepth( p_uDepth ), vecDescriptors( p_vecDescriptors ), matMask( p_vecMask )
    {
        //ds call recursive leaf spawner
        spawnLeafs( );
    }

    //ds only internally called
    CBNode( const uint64_t& p_uDepth,
            const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors,
            std::vector< uint32_t > p_vecSplitOrder ): uDepth( p_uDepth ), vecDescriptors( p_vecDescriptors )
    {
        //ds call recursive leaf spawner
        spawnLeafs( p_vecSplitOrder );
    }

    //ds only internally called
    CBNode( const uint64_t& p_uDepth,
            const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors,
            CDescriptorVector p_vecMask,
            const std::map< UIDLandmark, CBitStatistics >& p_mapBitStatistics ): uDepth( p_uDepth ), vecDescriptors( p_vecDescriptors ), matMask( p_vecMask )
    {
        //ds call recursive leaf spawner
        spawnLeafs( p_mapBitStatistics );
    }

public:

    ~CBNode( )
    {
        //ds nothing to do (the leafs will be freed manually)
    }

//ds fields
public:

    //ds rep
    const uint64_t uDepth;
    std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptors;
    int32_t uIndexSplitBit = -1;
    uint64_t uOnesTotal    = 0;
    bool bHasLeaves        = false;
    double dPartitioning   = 1.0;
    CDescriptorVector matMask;

    //ds info (incremented in tree during search)
    //uint64_t uLinkedPoints = 0;

    //ds peer: each node has two potential children
    CBNode* pLeafOnes  = 0;
    CBNode* pLeafZeros = 0;

//ds helpers
private:

    //ds helpers
    const double _getOnesFraction( const uint32_t& p_uIndexSplitBit, const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors, uint64_t& p_uOnesTotal ) const
    {
        assert( 0 < p_vecDescriptors.size( ) );

        //ds count
        uint64_t uNumberOfOneBits = 0;

        //ds just add the bits up (a one counts automatically as one)
        for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptor: p_vecDescriptors )
        {
            uNumberOfOneBits += cDescriptor.vecData[p_uIndexSplitBit];
        }

        //ds set total
        p_uOnesTotal = uNumberOfOneBits;
        assert( p_uOnesTotal <= p_vecDescriptors.size( ) );

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

    //ds returns a bitset with all bits set to true
    CDescriptorVector _getMaskClean( ) const
    {
        CDescriptorVector vecMask;
        vecMask.set( );
        return vecMask;
    }

    //ds filters multiple descriptors
    void _filterDescriptorsExhaustive( )
    {
        //ds unique descriptors (already add the front one first -> must be unique)
        std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsUNIQUE( 1, vecDescriptors.front( ) );

        //ds loop over current ones
        for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptor: vecDescriptors )
        {
            //ds check if matched
            bool bNotFound = true;

            //ds check uniques
            for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorUNIQUE: vecDescriptorsUNIQUE )
            {
                //ds if the actual descriptor is identical - and the key frame ID as well
                if( ( 0 == getDistanceHamming( cDescriptorUNIQUE.vecData, cDescriptor.vecData ) ) &&
                    ( cDescriptorUNIQUE.uIDKeyFrame == cDescriptor.uIDKeyFrame )                  )
                {
                    //ds already added to the unique vector - no further adding required
                    bNotFound = false;
                    break;
                }
            }

            //ds check if we failed to match the descriptor against the unique ones
            if( bNotFound )
            {
                vecDescriptorsUNIQUE.push_back( cDescriptor );
            }
        }

        assert( 0 < vecDescriptorsUNIQUE.size( ) );
        assert( vecDescriptorsUNIQUE.size( ) <= vecDescriptors.size( ) );

        //ds exchange internal version against unqiue
        vecDescriptors.swap( vecDescriptorsUNIQUE );
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

    //ds converts descriptors from cv::Mat to the current descriptor vector format
    inline static const Eigen::Matrix< double, uDescriptorSizeBits, 1 > getDescriptorVector( const CDescriptorVector& p_vecDescriptor )
    {
        //ds return vector
        Eigen::Matrix< double, uDescriptorSizeBits, 1 > vecDescriptor( Eigen::Matrix< double, uDescriptorSizeBits, 1 >::Zero( ) );

        //ds fill vector
        for( uint32_t u = 0; u < uDescriptorSizeBits; ++u )
        {
            if( p_vecDescriptor[u] )
            {
                vecDescriptor[u] = 1.0;
            }
        }

        return vecDescriptor;
    }

    //ds computes Hamming distance for Eigen::Matrix descriptors
    inline static const uint32_t getDistanceHamming( const CDescriptorVector& p_vecDescriptorQuery,
                                                     const CDescriptorVector& p_vecDescriptorReference )
    {
        //ds count set bits
        return ( p_vecDescriptorQuery ^ p_vecDescriptorReference ).count( );
    }

    //ds filters multiple descriptors
    inline static const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > getFilteredDescriptorsExhaustive( const std::vector< CDescriptorBRIEF< uDescriptorSizeBits > >& p_vecDescriptors )
    {
        //ds unique descriptors (already add the front one first -> must be unique)
        std::vector< CDescriptorBRIEF< uDescriptorSizeBits > > vecDescriptorsUNIQUE( 1, p_vecDescriptors.front( ) );

        //ds loop over current ones
        for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptor: p_vecDescriptors )
        {
            //ds check if matched
            bool bNotFound = true;

            //ds check uniques
            for( const CDescriptorBRIEF< uDescriptorSizeBits >& cDescriptorUNIQUE: vecDescriptorsUNIQUE )
            {
                //ds assuming key frame identity
                assert( cDescriptorUNIQUE.uIDKeyFrame == cDescriptor.uIDKeyFrame );

                //ds if the actual descriptor is identical - and the key frame ID as well
                if( 0 == CBNode< uMaximumDepth, uDescriptorSizeBits >::getDistanceHamming( cDescriptorUNIQUE.vecData, cDescriptor.vecData ) )
                {
                    //ds already added to the unique vector - no further adding required
                    bNotFound = false;
                    break;
                }
            }

            //ds check if we failed to match the descriptor against the unique ones
            if( bNotFound )
            {
                vecDescriptorsUNIQUE.push_back( cDescriptor );
            }
        }

        assert( 0 < vecDescriptorsUNIQUE.size( ) );
        assert( vecDescriptorsUNIQUE.size( ) <= p_vecDescriptors.size( ) );

        //ds exchange internal version against unqiue
        return vecDescriptorsUNIQUE;
    }

};

#endif //CBNODE_H
