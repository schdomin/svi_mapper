#ifndef CEXCEPTIONZERODISPARITY_H
#define CEXCEPTIONZERODISPARITY_H

#include <exception>

class CExceptionZeroDisparity: public std::exception
{

public:

    CExceptionZeroDisparity( const std::string& p_strExceptionDescription ): m_strExceptionDescription( p_strExceptionDescription )
    {

    }
    ~CExceptionZeroDisparity( )
    {

    }

private:

    const std::string m_strExceptionDescription;

public:

    virtual const char* what( ) const throw( )
    {
        return m_strExceptionDescription.c_str( );
    }
};

#endif //CEXCEPTIONZERODISPARITY_H
