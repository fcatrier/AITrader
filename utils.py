#
# Copyright (c) 2020-2021 by Frederi CATRIER - All rights reserved.
#

import numpy


def dict_get_param_list(dictionary):
    param_list = []
    for key, key_value in dictionary.items():
        param_list.append(key)
    return param_list


def dictionary_save(path, dictionary, postfix=''):
    #
    for key, key_value in dictionary.items():
        if postfix != '':
            numpy.save(path + '_hist_' + key + '_' + postfix + '.npy',  [key_value])
        else:
            numpy.save(path + '_hist_' + key + '.npy', [key_value])


def list_to_str(collection,separator):
   list_elements = ''
   first = True
   for element in collection:
      if(first==True):
         list_elements += str(element)
         first = False
      else:
         list_elements += separator
         list_elements += str(element)
   return list_elements
