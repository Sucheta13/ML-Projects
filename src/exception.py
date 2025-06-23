import sys
import logging

# Function is defined to show how the error message should be shown if error occurs inside a file
def error_message_detail(error,error_detail:sys):
    _,_,ex_tb=error_detail.exc_info()
    file_name=ex_tb.tb_frame.f_code.co_filename
    error_message="Error occured in the filename [{0}] in line no [{1}] with error message [{2}]".format(file_name,ex_tb.tb_lineno,str(error))

    return error_message


# Created a Custom_Exception which is inheriting Exception class and inheriting the __init__ and called the defines error_message_detail func to show the error in this way
class Custom_Exception(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail)

    def __str__(self):
        return self.error_message
