# FFS - Facial File Security
A simple file encryption/decryption software with an even simpler GUI.

## Goals
To add a Biometric measure to file encryption.

## About
FFS is a simple python application with less than 300 lines of code, that uses face recognition algorithms to encrypt/decrypt a file.

It uses the [haarcascade_frontalface_default.xml haarcascade file](https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml) to generate a model that is stored in a `.yml` file.

The yaml file is encrypted along with the file chosen by the user, and the key is stored in a `.key` file. The key can be accessed by the user using that file. Note that the key is to kept safely if the file is to be re-used. If you wish to simply encrypt the file and view it later, keep the `.key` file in the same location as it is created (or in the same directory as the script).

Decryption is relatively simple as well, you just need to select the file and if your face matches to that of the model stored in the yaml file, the file will decrypt else the script will throw a warning and quit.

## Technical details
Fernet encryption is used to encrypt and decrypt text files in this script.
For now, only a single file is supported, more files/filetypes is planned for the future.

Your face is simply used as a second method of authentication, this speciallly useful if you want to access secrets like your bitcoin wallet phase, or account recovery keys, etc.

This project is still heavily under development so do not try to encrypt very sensitive information, if you have anny suggestion however you can open a pull request and to ask doubts/questions, open an issue.

## Roadmap
- [x] - Encrypt and Decrypt files based on Face Recognition and Fernet encryption 
- [ ] - Multiple files support 
- [ ] - OCR for password entry instead of a `.key` file

## Credits
Face Recognition - https://github.com/thecodacus/Face-Recognition \
Cryptography - https://pypi.org/project/cryptography/   \
GUI - Tkinter
