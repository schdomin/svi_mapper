#ifndef CPDESCRIPTORBRIEF_H
#define CPDESCRIPTORBRIEF_H

#include <Eigen/Core>



template< uint32_t uDescriptorSizeBits = 256 >
struct CPDescriptorBRIEF
{
    //ds readability
    using CDescriptorVector = Eigen::Matrix< double, uDescriptorSizeBits, 1 >;

    CPDescriptorBRIEF( const uint64_t& p_uID,
                       const CDescriptorVector& p_vecData,
                       const uint64_t& p_uIDKeyFrame = 0 ): uID( p_uID ),
                                                            vecData( p_vecData ),
                                                            uIDKeyFrame( p_uIDKeyFrame )
    {
        //ds nothing to do
    }

    ~CPDescriptorBRIEF( )
    {
        //ds nothing to do
    }

    const uint64_t uID;
    const CDescriptorVector vecData;
    const uint64_t uIDKeyFrame;
};

#endif //CPDESCRIPTORBRIEF
