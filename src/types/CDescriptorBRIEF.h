#ifndef CDESCRIPTORBRIEF_H
#define CDESCRIPTORBRIEF_H

#include <bitset>

//#define SHUFFLE_DESCRIPTORS



template< uint32_t uDescriptorSizeBits = 256 >
struct CDescriptorBRIEF
{
    //ds readability
    using CDescriptorVector = std::bitset< uDescriptorSizeBits >;

    CDescriptorBRIEF( const uint64_t& p_uID,
                      const CDescriptorVector& p_vecData,
                      const uint64_t& p_uIDKeyFrame = 0,
                      const uint64_t& p_uIDLandmark = 0 ): uID( p_uID ),
                                                           vecData( p_vecData ),
                                                           uIDKeyFrame( p_uIDKeyFrame ),
                                                           uIDLandmark( p_uIDLandmark )
    {
        //ds nothing to do
    }

    ~CDescriptorBRIEF( )
    {
        //ds nothing to do
    }

#ifndef SHUFFLE_DESCRIPTORS

    const uint64_t uID;
    const CDescriptorVector vecData;
    const uint64_t uIDKeyFrame;
    const uint64_t uIDLandmark;

#else

    uint64_t uID;
    CDescriptorVector vecData;
    uint64_t uIDKeyFrame;
    uint64_t uIDLandmark;

#endif

};

#endif //CDESCRIPTORBRIEF
