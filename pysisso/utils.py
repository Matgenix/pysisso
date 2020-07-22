# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL


import os
import stat
import tempfile
import monty.shutil


# class FakeExec:
#
#     def __init__(self, exec_name, remove_temp_bin_dir=True):
#         self.exec_name = exec_name
#         self.remove_temp_bin_dir = remove_temp_bin_dir
#
#     def __enter__(self):
#         self.temp_bin_dir = tempfile.mkdtemp(prefix='bin', dir='.')
#         # Create fake executable file and make it executable (equivalent of "chmod +x")
#         exec_path = os.path.join(self.temp_bin_dir, self.exec_name)
#         print(exec_path)
#         with open(exec_path, 'w') as f:
#             f.write('#! /usr/bin/env python\npass\n')
#         st = os.stat(exec_path)
#         os.chmod(exec_path, st.st_mode | stat.S_IEXEC)
#
#         # Update PATH
#         self.current_OS_PATH = os.environ.get('PATH', None)
#         if self.current_OS_PATH is None:
#             paths = []
#         else:
#             paths = self.current_OS_PATH.split(':')
#         paths.insert(0, self.temp_bin_dir)
#         os.environ['PATH'] = ':'.join(paths)
#
#         return exec_path
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # Remove temporary bin directory with fake executable from system's PATH environment variable
#         OS_PATH = os.environ['PATH']
#         paths = OS_PATH.split(':')
#         paths.remove(self.temp_bin_dir)
#         os.environ['PATH'] = ':'.join(paths)
#
#         if self.remove_temp_bin_dir:
#             monty.shutil.remove(self.temp_bin_dir)
