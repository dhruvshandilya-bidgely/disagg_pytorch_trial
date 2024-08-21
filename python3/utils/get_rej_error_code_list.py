"""
Author - Mayank Sharan
Date - 17/03/20
Convert the list of rejection reasons to list of corresponding error codes
"""

# Import functions from within the project

from python3.config.mappings.error_code_mapping import ErrorCodeMapping


def get_rej_error_code_list(rejection_reasons):

    """
    Utility function to convert a string list of rejection reasons to corresponding error codes

    Parameters:
        rejection_reasons           (list)              : List of rejection reasons for pipeline not running

    Returns:
        error_code_list             (list)              : List of error codes corresponding to the rejection reasons
    """

    # THIS FUNCTION IS TO BE ONLY USED FOR DATA QUALITY REJECTION

    error_code_list = []

    for reason in rejection_reasons:

        error_code = ErrorCodeMapping.ERROR_CODE_DICT.get(reason)

        if error_code is not None:
            error_code_list.append(int(error_code))

    return error_code_list
