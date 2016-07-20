#!/usr/bin/env python
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# recognition

#trying argument parsing !!!!



"""
This script uses the Vision API's OCR capabilities to find and index any text
a set of images. It builds an inverted index, and uses redis
(http://redis.io/) to persist the index. By default, the script asumes a local
redis install set up to persist to disk. Assuming the redis database is
persisted between runs, the script can be run multiple times on the same set
of files without redoing previous work. The script uses also nltk
(http://www.nltk.org/index.html) to do stemming and tokenizing.

To run the example, install the necessary libraries by running:

    pip install -r requirements.txt

Then, follow the instructions here:
http://www.nltk.org/data.html to download the necessary nltk data.

Run the script on a directory of images to create the index, E.g.:

    ./textindex.py <path-to-image-directory>

Then, instantiate an instance of the Index() object (via a script or the
Python interpreter) and use it to look up words via the Index.lookup() or
Index.print_lookup() methods.  E.g.:

    import textindex
    index = textindex.Index()
    index.print_lookup('cats', 'dogs')

This will return all the images that include both 'cats' and 'dogs' in
recognizable text. More exactly, it will return all images that include text
with the same stems.
"""

import argparse
# [START detect_text]
import base64
import os
import re
import sys
from PIL import Image, ImageEnhance
import time
import glob
import tempfile
import shutil

from resizeimage import resizeimage

import tempfile

from googleapiclient import discovery
from googleapiclient import errors
import nltk
from nltk.stem.snowball import EnglishStemmer
from oauth2client.client import GoogleCredentials
import redis
from scipy.misc import imsave


DISCOVERY_URL = 'https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'  # noqa
BATCH_SIZE = 10


class VisionApi:
    """Construct and use the Google Vision API service."""

    def __init__(self, api_discovery_file='vision_api.json'):
        self.credentials = GoogleCredentials.get_application_default()
        self.service = discovery.build(
            'vision', 'v1', credentials=self.credentials,
            discoveryServiceUrl=DISCOVERY_URL)

    def detect_text(self, input_filenames, num_retries=3, max_results=6):
        """Uses the Vision API to detect text in the given file.
        """
        images = {}
        for filename in input_filenames:
            with open(filename, 'rb') as image_file:
                images[filename] = image_file.read()

        batch_request = []
        for filename in images:
            batch_request.append({
                'image': {
                    'content': base64.b64encode(
                            images[filename]).decode('UTF-8')
                },
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': max_results,
                }]
            })
        request = self.service.images().annotate(
            body={'requests': batch_request})

        try:
            responses = request.execute(num_retries=num_retries)
            if 'responses' not in responses:
                return {}
            text_response = {}
            for filename, response in zip(images, responses['responses']):
                if 'error' in response:
                    print("API Error for %s: %s" % (
                            filename,
                            response['error']['message']
                            if 'message' in response['error']
                            else ''))
                    continue
                if 'textAnnotations' in response:
                    text_response[filename] = response['textAnnotations']
                else:
                    text_response[filename] = []
            return text_response
        except errors.HttpError as e:
            print("Http Error for %s: %s" % (filename, e))
        except KeyError as e2:
            print("Key error: %s" % e2)
# [END detect_text]


# The inverted index is based in part on this example:
# http://tech.swamps.io/simple-inverted-index-using-nltk/
class Index:
    """ Inverted index datastructure """

    def __init__(self, tokenizer=nltk.word_tokenize,
                 stemmer=EnglishStemmer(),
                 stopwords=nltk.corpus.stopwords.words('english')):
        """Create an inverted index.

        Args:
          tokenizer -- NLTK compatible tokenizer function
          stemmer   -- NLTK compatible stemmer
          stopwords   -- list of ignored words

        This code assumes that a local redis server is running, and assumes
        that you're not already using 'db0' and 'db1' of that installation
        for some other purpose. Change these client calls if necessary for
        your redis config.
        """

        # db 0 holds the token (words) inverted index.
        self.redis_token_client = redis.StrictRedis(db=0)
        # db 1 holds the filename--> text mapping.
        self.redis_docs_client = redis.StrictRedis(db=1)
        # Do an initial check on the redis connection. If redis is not up,
        # the constructor call will fail.
        self.redis_docs_client.ping()
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.__unique_id = 0
        self.stopwords = set(stopwords) if stopwords else set()

    def lookup(self, *words):
        """Look up words in the index; return the intersection of the hits."""
        conjunct = set()

        for word in words:
            word = word.lower()

            if self.stemmer:
                word = self.stemmer.stem(word)

            docs_with_word = self.redis_token_client.smembers(word)
            hits = set([
                (id, self.redis_docs_client.get(id))
                for id in docs_with_word
            ])
            conjunct = conjunct & hits if conjunct else hits

        return conjunct

    def print_lookup(self, *words):
        """Print lookup results to stdout."""
        hits = self.lookup(*words)
        if not hits:
            print("No hits found.")
            return
        for i in hits:
            print("***Image %s has text:\n%s" % i)

##    def document_is_processed(self, filename):
##        """Check whether a document (image file) has already been processed.
##        """
##        res = self.redis_docs_client.get(filename)
##        if res:
##            print("%s already added to index." % filename)
##            return True
##        if res == '':
##            print('File %s was already checked, and contains no text.'
##                  % filename)
##            return True
##        return False

    def set_contains_no_text(self, filename):
        """Add bookkeeping to indicate that the given file had no
        discernible text."""
        self.redis_docs_client.set(filename, '')

    def add(self, filename, document):
        """
        Add a document string to the index.
        """
        # You can uncomment the following line to see the words found in each
        # image.
        print("WORDS FOUND IN SAMPLE IMAGE %s: \n%s" % (filename, document))
        print('\n')
        for token in [t.lower() for t in nltk.word_tokenize(document)]:
            if token in self.stopwords:
                continue
            if token in ['.', ',', ':', '']:
                continue
            if self.stemmer:
                token = self.stemmer.stem(token)
            # Add the filename to the set associated with the token.
            self.redis_token_client.sadd(token, filename)

        # store the 'document text' for the filename.
        self.redis_docs_client.set(filename, document)
        #print document

    def delete_all_in_index(self):
    # index.get_range by returns up to 100 documents at a time, so we must
    # loop until we've deleted all items.
        while True:
        # Use ids_only to get the list of document IDs in the index without
        # the overhead of getting the entire document.
            document_ids = [
                document.doc_id
                for document
                in index.get_range(ids_only=True)]
    
            # If no IDs were returned, we've deleted everything.
            if not document_ids:
                break
    
            # Delete the documents for the given IDs
            index.delete(document_ids)




##        print('DOES THE REFERENCE IMAGE TEXT MATCH WITH THE TEXT IN THE SAMPLE IMAGE?')
##        f = open('/Users/user/Documents/refdoc.txt', 'r')
##        index = 0
##        count = 0
##        for line in f:
##            index = index + 1 
##            line = line.replace(" ","")
##            line = line.rstrip('\n')
##            document = document.replace(" ","")
##            if ((line.lower()) in (document.lower())):
##                count = count + 1
##        if (index==count):
##            print " YES ! Store followed the instructions properly"
##        else:
##            print " NO :("





        
def get_words(text):
    return re.compile('\w+').findall(text)


# [START extract_descrs]
def extract_description(texts):
    """Returns all the text in text annotations as a single string"""
    document = ''
    for text in texts:
        try:
            document += text['description']
        except KeyError as e:
            print('KeyError: %s\n%s' % (e, text))
    return document


def extract_descriptions(input_filename, index, texts):
    """Gets and indexes the text that was detected in the image."""


    print('\n')
    print ('READING SAMPLE IMAGE %s..'% input_filename)
    print('\n')

    
    if texts:
        document = extract_description(texts)
        index.add(input_filename, document)
        sys.stdout.write('.')  # Output a progress indicator.
        sys.stdout.flush()
    else:
        if texts == []:
            print('%s had no discernible text.' % input_filename)
            index.set_contains_no_text(input_filename)




    return document




# [END extract_descrs]


# [START get_text]
def get_text_from_files(vision, index, input_filenames):
    """Call the Vision API on a file and index the results."""
    document3 = ""
    texts = vision.detect_text(input_filenames)
    for filename, text in texts.items():
        document = extract_descriptions(filename, index, text)
        document3 = document3 + document




    return document3


def batch(iterable, batch_size=BATCH_SIZE):
    """Group an iterable into batches of size batch_size.

    >>> tuple(batch([1, 2, 3, 4, 5], batch_size=2))
    ((1, 2), (3, 4), (5))
    """
    b = []
    for i in iterable:
        b.append(i)
        if len(b) == batch_size:
            yield tuple(b)
            b = []
    if b:
        yield tuple(b)

def main(input_dir):
    """Walk through all the not-yet-processed image files in the given
    directory, extracting any text from them and adding that text to an
    inverted index.
    """

    document_add = ""
    # Create a client object for the Vision API
    vision = VisionApi()
    # Create an Index object to build query the inverted index.
##    sucess = False
##    count = 0
##    while (sucess==False & count<=5):
##        try:
    index = Index()
##            sucess = True
##        except:
##            print "wow"
##            count = count + 1
##            time.sleep(6)
            

    allfileslist = []
    # Recursively construct a list of all the files in the given input
    # directory.
    for folder, subs, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                allfileslist.append(os.path.join(folder, filename))

    fileslist = []
    for filename in allfileslist:
        # Look for text in any files that have not yet been processed.
##        if index.document_is_processed(filename):
##            continue
        fileslist.append(filename)

    for filenames in batch(fileslist):
        document = get_text_from_files(vision, index, filenames)


##    for filenames in filee:
##    document = get_text_from_files(vision,index,filee)
##
##
##    print document
    return document
# [END get_text]

if __name__ == '__main__':
##    parser = argparse.ArgumentParser(
##        description='Detects text in the images in the given directory.')
##    parser.add_argument(
##        'input_directory',
##        help='the image directory you\'d like to detect text in.')
##    args = parser.parse_args()

##    main(args.input_directory)
##    document1 = main('/Users/user/Documents/images210')
##    print document1
####
####
##    image = Image.open('/Users/user/Documents/images210/hard.jpg')
##    contrast = ImageEnhance.Contrast(image)
##    image.show()
##
##    contrast.enhance(2).show()
##    contrast.enhance(2).save('/Users/user/Documents/imagesa210/hard1.jpg')
##
##    document2 = main('/Users/user/Documents/imagesa210')
##    document3 = document1 + document2
##    print document3
##

##    image_file = Image.open("/Users/user/Documents/imagesa223/sale1.jpg") # open colour image
##
##    image_file= image_file.convert('1') # convert image to black and white
##    imsave('/Users/user/Documents/imagesa223/result_col.png', image_file)
##
##    
##    with open('/Users/user/Documents/pizza2/pizza.jpg', 'r+b') as f:
##        with Image.open(f) as image:
##            width,height = image.size
##            cover = resizeimage.resize_cover(image, [width/2, height/2],validate=False)
##            cover.save('/Users/user/Documents/pizza1/pizza_resize50.jpg', image.format)
##            if(((width/4)!=0)and((height/4)!=0)):
##                cover = resizeimage.resize_cover(image, [width/4, height/4],validate=False)
##                cover.save('/Users/user/Documents/pizza1/pizza_resize25.jpg', image.format)
####

##      dirpath = tempfile.mkdtemp()
##      print dirpath
    # ... do stuff with dirpath
      parser = argparse.ArgumentParser()
      parser.add_argument("image_path", type=str,action = 'store')
      parser.add_argument("filename",action = 'store')
      args = parser.parse_args()
      col = Image.open(args.image_path)
      td = tempfile.gettempdir()
      col.save(td+'/1.jpg', 'JPEG')


 ##     col.save("dirpath/1.jpg")
  


      gray = col.convert('L')
      bw = gray.point(lambda x: 0 if x<128 else 255, '1')
      bw.save(td+'/result.jpg','JPEG')
##      bw.save("dirpath")

    
##      with open('/Users/user/Documents/high369/1.jpg', 'r+b') as f:
##          with Image.open(f) as image:
##              width,height = image.size
##              if((width/2!=0)and(height/2!=0)):
##                  image = image.resize((int(width/2), int(height/2)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high369/1_2.jpg', 'JPEG', quality=quality_val)
##
##              if((width/4!=0)and(height/4!=0)):
##                  image = image.resize((int(width/4), int(height/4)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high369/1_4.jpg', 'JPEG', quality=quality_val)
##
##              if((width/6!=0)and(height/6!=0)):
##                  image = image.resize((int(width/6), int(height/6)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high369/1_6.jpg', 'JPEG', quality=quality_val)
##
##              if((width/8!=0)and(height/8!=0)):
##                  image = image.resize((int(width/8), int(height/8)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high369/1_8.jpg', 'JPEG', quality=quality_val)
##
##              if((width/10!=0)and(height/10!=0)):
##                  image = image.resize((int(width/10), int(height/10)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high34/1_10.jpg', 'JPEG', quality=quality_val)
##
##              if((width/12!=0)and(height/12!=0)):
##                  image = image.resize((int(width/12), int(height/12)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high369/1_12.jpg', 'JPEG', quality=quality_val)
##
##              if((width/15!=0)and(height/15!=0)):
##                  image = image.resize((int(width/15), int(height/15)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high369/1_15.jpg', 'JPEG', quality=quality_val)
##
##
##      with open('/Users/user/Documents/high369/result.jpg', 'r') as f:
##          with Image.open(f) as image:
##              width,height = image.size
##              if((width/2!=0)and(height/2!=0)):
##                  image = image.resize((int(width/2), int(height/2)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high369/result_2.jpg', 'JPEG', quality=quality_val)
##
##              if((width/4!=0)and(height/4!=0)):
##                  image = image.resize((int(width/4), int(height/4)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high369/result_4.jpg', 'JPEG', quality=quality_val)
##
##              if((width/6!=0)and(height/6!=0)):
##                  image = image.resize((int(width/6), int(height/6)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high369/result_6.jpg', 'JPEG', quality=quality_val)
##
##              if((width/8!=0)and(height/8!=0)):
##                  image = image.resize((int(width/8), int(height/8)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high369/result_8.jpg', 'JPEG', quality=quality_val)
##
##              if((width/10!=0)and(height/10!=0)):
##                  image = image.resize((int(width/10), int(height/10)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high369/result_10.jpg', 'JPEG', quality=quality_val)
##
##              if((width/12!=0)and(height/12!=0)):
##                  image = image.resize((int(width/12), int(height/12)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high369/result_12.jpg', 'JPEG', quality=quality_val)
##
##              if((width/15!=0)and(height/15!=0)):
##                  image = image.resize((int(width/15), int(height/15)), Image.ANTIALIAS)
##                  quality_val = 90
##                  image.save('/Users/user/Documents/high369/result_15.jpg', 'JPEG', quality=quality_val)
##

      document2 = main(td)
##      document2 = main("/Users/user/Desktop/images")
      document2 = document2.replace('\n', '')
      document2 = document2.replace("(", "")
      document2 = document2.replace("/", "")
      document2 = document2.replace(".", "")
      document2 = document2.replace("?", "")
      document2 = document2.replace("<", "")
      document2 = document2.replace(">", "")
      document2 = document2.replace(")", "")
      document2 = document2.replace(" ", "")
    
      ##print document2

      print('DOES THE REFERENCE IMAGE TEXT MATCH WITH THE TEXT IN THE SAMPLE IMAGE?')

 
      with open(args.filename) as file:
          f = open(args.filename) 
          index = 0
          count = 0
    
          document2 = document2.decode('utf-8')
 
          for line in f:
               index = index + 1 
               line = line.replace(" ","")
               line = line.rstrip('\n')
               print line
               if ((line.lower()) in (document2.lower())):
                   count = count + 1
          if (index==count):
               print " YES ! Store followed the instructions properly"
          else:
               print " NO :("

      shutil.rmtree(td)
      
    
##          files = glob.glob('/Users/user/Desktop/images/*')
##          for f in files:
##                os.remove(f)

##      def deleteAllInIndex(cls):
##        """Delete all the docs in the given index."""
##        docindex = cls.getIndex()
##  
##        try:
##          while True:
##            # until no more documents, get a list of documents,
##            # constraining the returned objects to contain only the doc ids,
##            # extract the doc ids, and delete the docs.
##            document_ids = [document.doc_id
##                            for document in docindex.list_documents(ids_only=True)]
##            if not document_ids:
##              break
##            docindex.remove(document_ids)
##        except search.Error:
##          logging.exception("Error removing documents:")
               
## 
