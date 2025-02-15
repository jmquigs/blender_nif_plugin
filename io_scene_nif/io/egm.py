"""This module is used to for Nif file operations"""

# ***** BEGIN LICENSE BLOCK *****
#
# Copyright © 2016, NIF File Format Library and Tools contributors.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above
#      copyright notice, this list of conditions and the following
#      disclaimer in the documentation and/or other materials provided
#      with the distribution.
#
#    * Neither the name of the NIF File Format Library and Tools
#      project nor the names of its contributors may be used to endorse
#      or promote products derived from this software without specific
#      prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# ***** END LICENSE BLOCK *****


from pyffi.formats.egm import EgmFormat
from io_scene_nif.utility.nif_logging import NifLog
from io_scene_nif.utility.nif_utils import NifError

class EGMFile():
    """Load and save a FaceGen Egm file"""

    @staticmethod
    def load_egm(file_path):
        """Loads an egm file from the given path"""
        NifLog.info("Loading {0}".format(file_path))
        
        egm_file = EgmFormat.Data()
        
        # open keyframe file for binary reading
        with open(file_path, "rb") as egm_stream:
            # check if nif file is valid
            egm_file.inspect_quick(egm_stream)
            if egm_file.version >= 0:
                # it is valid, so read the file
                NifLog.info("EGM file version: {0}".format(egm_file.version, "x"))
                NifLog.info("Reading FaceGen egm file")
                egm_file.read(egm_stream)
            elif egm_file.version == -1:
                raise NifError("Unsupported EGM version.")
            else:                    
                raise NifError("Not a EGM file.")
            
        return egm_file
    