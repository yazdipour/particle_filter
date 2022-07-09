# -*- coding: utf-8 -*-
"""Collection of classes and methods to generate human-readable test output"""
from __future__ import absolute_import, print_function

import sys

import six
from six.moves.queue import LifoQueue

SUITETAG = "[" + "<" * 77 + "]"
SUITEBREAKTAG = "[" + "-" * 77 + "]"
CASETAG = "[" + "<" * 39 + "]"
CASEOPENTAG = "[" + "=" * 39 + "]"
CASEBREAKTAG = "[" + "-" * 39 + "]"
TESTTAG = "[" + "=" * 77 + "]"
TESTTAG_SHORT = "[" + "-" * 39 + "]"
EMPTY = "[" + " " * 77 + "]"
EMPTY_SHORT = "[" + " " * 39 + "]"
RUN = "[  RUN     ]"
INFO = "[  INFO    ]"
OK = "[      OK  ]"
FAILED = "[  FAILED  ]"
PASSED = "[  PASSED  ]"
CASE = "Case"
SUITE = "Suite"
TEST = "Test"
VALIDATION = "Validation"
TESTSUITE = TEST + "-" + SUITE
TESTCASE = TEST + "-" + CASE
VALIDATIONSUITE = VALIDATION + "-" + SUITE
VALIDATIONCASE = VALIDATION + "-" + CASE


class TestSuite(object):
    """
    Templatelike suite of strings for testing purposes. Calls the internal
    suite_open(info) function.

    Attributes
    ----------
    name : str
        The name of the test suite.
    lifo : LifoQueue
        a last in first out (lifo) queue for subsequent tests.

    Parameters
    ----------
    name : str
        The name of the test suite.
    info : str or list of str
        A string or list of string which will be printed on start.
    """
    def __init__(self, name, info=None):
        self.name = name
        self.suite_open(info)
        self.lifo_test = LifoQueue()
        self.lifo_case = LifoQueue()

    def suite_open(self, info=None):
        """
        Prints the opening of a test suite.

        Parameters
        ----------
        info : str (default: None)
            A string or list of string which will be printed.
        """
        print(SUITETAG)
        print(RUN, self.name, TESTSUITE)
        if info is not None:
            for string in info:
                print(INFO, string)

    @staticmethod
    def suite_info(msg):
        """Print a message with the INFO tag"""
        print(INFO, msg)

    def suite_close(self, passed):
        """
        Prints the closing of the test case.

        Parameters
        ----------
        passed : boolean
            Determines whether the passed or failed message is printed.
        """
        print(SUITEBREAKTAG)
        if passed:
            print(PASSED, self.name, TESTSUITE)
        else:
            print(FAILED, self.name, TESTSUITE)
        print(SUITETAG)

    def testcase_open(self, name, info=None):
        """
        Prints the opening of a test case.

        Parameters
        ----------
        info : str (default: None)
            A string or list of string which will be printed.
        """
        self.lifo_case.put(name)
        print(CASETAG)
        print(RUN, name, TESTCASE)
        if info is not None:
            for string in info:
                print(INFO, string)
        print(CASEOPENTAG)

    def testcase_close(self, passed):
        """
        Prints the closing of the test suite.

        Parameters
        ----------
        passed : boolean
            Determines whether the passed or failed message is printed.
        """
        name = self.lifo_case.get()
        print(CASEBREAKTAG)
        if passed:
            print(OK, name, TESTCASE)
        else:
            print(FAILED, name, TESTCASE)
        print(CASETAG)

    def test_open(self, name, info=None):
        """
        Prints the opening of a test.

        Parameters
        ----------
        name : str
            The name of the test (will be buffered in the lifo queue)
        info : str (default: None)
            A string or list of string which will be printed.
        """
        self.lifo_test.put(name)
        if self.lifo_case.empty():
            print(TESTTAG)
        else:
            print(TESTTAG_SHORT)
        print(RUN, name, TEST)
        if info is not None:
            if isinstance(info, (six.string_types, six.text_type, six.binary_type)):
                print(INFO, info)
            else:
                for msg in info:
                    print(INFO, msg)

    def test_close(self, passed, error_msg=None):
        """
        Prints the closing of a test.

        Parameters
        ----------
        passed : boolean
            Determines whether the passed or failed message is printed.
        error_msg : str (default: None)
            A string or list of string which will be printed when a failure
            occurred.
        """
        name = self.lifo_test.get()
        if self.lifo_case.empty():
            print(TESTTAG)
        else:
            print(TESTTAG_SHORT)
        if passed:
            print(OK, name, TEST)
        else:
            if error_msg is not None:
                if isinstance(error_msg, (six.string_types, six.text_type, six.binary_type)):
                    print(error_msg)
                else:
                    for msg in error_msg:
                        print(msg)
            print(FAILED, name, TEST)
        if self.lifo_case.empty():
            print(TESTTAG)
        else:
            print(TESTTAG_SHORT)

    @staticmethod
    def test_oneline(passed, name, error_msg=None):
        """
        Prints the result of a one line test.

        Parameters
        ----------
        passed : boolean
            Determines whether the passed or failed message is printed.
        name : str
            The name of the test.
        error_msg : str (default: None)
            A string or list of string which will be printed when a failure
            occurred.
        """
        if passed:
            print(OK, name, TEST)
        else:
            print(FAILED, name, TEST)
            if error_msg is not None:
                print(error_msg)


class ValidationSuite(object):
    """
    Templatelike suite of strings for validation purposes. Calls the internal
    suite_open(info) function.

    Attributes
    ----------
    name : str
        The name of the validation suite.
    lifo : LifoQueue
        a last in first out (lifo) queue for subsequent validations.

    Parameters
    ----------
    name : str
        The name of the validation suite.
    info : str or list of str
        A string or list of string which will be printed on start.
    """
    def __init__(self, name, info=None):
        self.name = name
        self.suite_open(info)
        self.lifo = LifoQueue()
        self.lifo_case = LifoQueue()

    def suite_open(self, info):
        """
        Prints the opening of a validation suite.

        Parameters
        ----------
        info : str (default: None)
            A string or list of string which will be printed.
        """
        print(SUITETAG)
        print(RUN, self.name, VALIDATIONSUITE)
        if info is not None:
            for string in info:
                print(INFO, string)

    @staticmethod
    def suite_info(msg):
        """Print a message with the INFO tag"""
        print(INFO, msg)

    def suite_close(self, passed):
        """
        Prints the closing of the validation suite.

        Parameters
        ----------
        passed : boolean
            Determines whether the passed or failed message is printed.
        """
        print(SUITEBREAKTAG)
        if passed:
            print(PASSED, self.name, VALIDATIONSUITE)
        else:
            print(FAILED, self.name, VALIDATIONSUITE)
        print(SUITETAG)

    def validation_open(self, name, info=None):
        """
        Prints the opening of a validation.

        Parameters
        ----------
        name : str
            The name of the test (will be buffered in the lifo queue)
        info : str (default: None)
            A string or list of string which will be printed.
        """
        self.lifo.put(name)
        print("[" + "=" * 77 + "]")
        print(RUN, name, VALIDATION)
        if info is not None:
            for string in info:
                print(INFO, string)

    def validation_close(self, passed, error_msg=None):
        """
        Prints the closing of a validation.

        Parameters
        ----------
        passed : boolean
            Determines whether the passed or failed message is printed.
        error_msg : str (default: None)
            A string or list of string which will be printed when a failure
            occurred.
        """
        name = self.lifo.get()
        if passed:
            print(OK, name, VALIDATION)
        else:
            if error_msg is not None:
                print(error_msg)
            print(FAILED, name, VALIDATION)

    @staticmethod
    def validation_oneline(passed, name, error_msg=None):
        """
        Prints the result of a one line test.

        Parameters
        ----------
        passed : boolean
            Determines whether the passed or failed message is printed.
        name : str
            The name of the test.
        error_msg : str (default: None)
            A string or list of string which will be printed when a failure
            occurred.
        """
        if passed:
            print(OK, name, VALIDATION)
        else:
            if error_msg is not None:
                print(error_msg)
            print(FAILED, name, VALIDATION)

    def validationcase_open(self, name, info=None):
        """
        Prints the opening of a test case.

        Parameters
        ----------
        info : str (default: None)
            A string or list of string which will be printed.
        """
        self.lifo_case.put(name)
        print(CASETAG)
        print(RUN, name, VALIDATIONCASE)
        if info is not None:
            for string in info:
                print(INFO, string)
        print(CASEOPENTAG)

    def validationcase_close(self, passed):
        """
        Prints the closing of the test suite.

        Parameters
        ----------
        passed : boolean
            Determines whether the passed or failed message is printed.
        """
        name = self.lifo_case.get()
        print(CASEBREAKTAG)
        if passed:
            print(OK, name, VALIDATIONCASE)
        else:
            print(FAILED, name, VALIDATIONCASE)
        print(CASETAG)


def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    """
    Print iterations progress

    This implementation is based on
    https://gist.github.com/aubricus/f91fb55dc6ba5557fbab06119420dd6a

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int((bar_length * iteration / float(total)) + 0.5)
    bar = u'0' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write(u'\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix))

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()
