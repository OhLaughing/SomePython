from ftplib import FTP
import os

class FTPservice(object):
    def __init__(self, host, port, username, password):
        self.__host = host
        self.__port = port
        self.__username = username
        self.__password = password
        self.__ftp = FTP()

    def connect_to_ftp(self):
        self.__ftp.connect(self.__host, self.__port)
        self.__ftp.login(self.__username, self.__password)
        print self.__ftp.getwelcome()

    def disconnect_from_ftp(self):
        self.__ftp.close()

    def list_all_files_of_path(self, path):
        return self.__ftp.nlst(path)

    def check_if_path_exist(self, path):
        if path.startswith('/'):
            self.__ftp.cwd('/')
            path = path[1:-1]
        paths = path.split('/')

    def change_directory(self, path):
        self.__ftp.cwd(path)

    def current_working_directory(self):
        return self.__ftp.pwd()

    def determin_if_is_a_file(self, dir):
        files = self.__ftp.nlst();
        if dir not in files:
            return False
        try:
            self.__ftp.cwd(dir)
        except Exception:
            return True
        return False

    def make_directory(self, folder_name):
        self.__ftp.mkd(folder_name)

    def if_path_exist(self, path):
        if path.startswith('/'):
            self.__ftp.cwd('/')
            path = path[1:len(path)]
        paths = path.split("/")
        for p in paths:
            files = self.__ftp.nlst()
            if p not in files:
                return False
            self.__ftp.cwd(p)
        print 'current working directory: ' + self.__ftp.pwd()
        return True

    def upload_file(self, local_file):
        '''
        should change work directory first
        '''
        print 'local file name=', local_file

        if not os.path.exists(local_file) or not os.path.isfile(local_file):
            cwd = os.getcwd()
            print 'current working directory is:', cwd

            if 'RfAutoTest' not in cwd:
                pass
            else:
                pos = cwd.index('RfAutoTest')
                rf_dir = os.path.join(cwd[:pos], 'RfAutoTest')
                local_file = os.path.join(os.path.join(rf_dir, 'data\\sdrommb\\ddm'), local_file)

        file_handler = open(local_file, 'rb')
        file_name = os.path.basename(local_file)

        try:
            print self.__ftp.storbinary('STOR ' + file_name, file_handler)

            return True
        except Exception, e:
            raise Exception('upload file exception:\n' + str(e))
        finally:
            file_handler.close()

    def download_file(self, remote_file, local_file):
        file_handler = open(local_file, 'wb')

        try:
            self.__ftp.retrbinary('RETR ' + remote_file, file_handler.write)

            return True
        except Exception, e:
            raise Exception('download file exception:\n' + str(e))
        finally:
            file_handler.close()

    def delete_file(self, part_name):
        """
        should change working directory to the folder that contains the file
        :param part_name:
        :return:
        """
        for f in self.__ftp.nlst():
            if part_name in f:
                try:
                    self.__ftp.delete(f)
                except Exception as e:
                    print 'WARN: delete file [' + f + '] failed:', e
