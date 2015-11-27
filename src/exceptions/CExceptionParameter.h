#ifndef CEXCEPTIONPARAMETER_H_
#define CEXCEPTIONPARAMETER_H_

#include <exception>

class CExceptionParameter: public std::exception
{

public:

    CExceptionParameter( const std::string& p_strExceptionDescription ): m_strExceptionDescription( p_strExceptionDescription )
    {

    }
    ~CExceptionParameter( )
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

#endif //CEXCEPTIONPARAMETER_H_
