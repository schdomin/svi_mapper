#ifndef CDESCRIPTORBRIEF_H
#define CDESCRIPTORBRIEF_H

#include <Eigen/Core>



template< uint32_t uDescriptorSizeBits = 256 >
struct CDescriptorBRIEF
{
    //ds readability
    using CDescriptorVector = Eigen::Matrix< bool, uDescriptorSizeBits, 1 >;

    //ds memory alignment
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CDescriptorBRIEF(  const uint64_t& p_uID, const CDescriptorVector& p_vecData ): uID( p_uID ), vecData( p_vecData )
    {
        //ds nothing to do
    }

    ~CDescriptorBRIEF( )
    {
        //ds nothing to do
    }

    const uint64_t uID;
    const CDescriptorVector vecData;
};

#endif //CDESCRIPTORBRIEF
