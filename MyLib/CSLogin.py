import cryosparc.tools as cst
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import MyLib.mytoolbox as mytoolbox


class CryoSPARC_Login():
    def __init__(self, license, host, port):
        self.__LICENSE = license
        self.__HOST = host
        self.__PORT = port

    def GetCryoSPARCHandle(self, email, password):
        cshandle = cst.CryoSPARC(license=self.__LICENSE, email=email, password=password, host=self.__HOST, base_port=self.__PORT)
        return cshandle


cs_login_info = mytoolbox.readjson(os.path.dirname(__file__) + '/cs_login_info.json')
cshandleclass = CryoSPARC_Login(license=cs_login_info['license'], host=cs_login_info['host'], port=cs_login_info['port'])