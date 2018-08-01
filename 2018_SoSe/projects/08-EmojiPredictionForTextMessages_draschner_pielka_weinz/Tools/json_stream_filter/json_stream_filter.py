#!/usr/bin/env python3

import fileinput as fi
import json
import sys
import re

JSON_INDENT=2

# a little helper tool for processing and manipulating json streams.
# just because jq is a little bit limited in its abilities to process json values.
# especially here to filter out emojis with regexp and put filtered elements into new key-value pairs

class regex_tuple(object):
    '''
    a regex tuple object consists of:
      * regexp_str: regexp string
      * keys_to_process: keys which are processed
      * key_to_store_match
    '''
    def __init__(self, regexp_str, keys_to_process, replace_str = None, key_to_store_match = None):
        self.regexp_str = regexp_str
        self.keys_to_process = keys_to_process
        self.key_to_store_match = key_to_store_match
        self.replace_str = replace_str
        self.regexp_obj = re.compile(self.regexp_str)
    
    @staticmethod
    def create_from_string(s):
        # delimiter is the first char:
        d = s[0]
        vals = s[1:].split(d)
        if len(vals) != 4:
            sys.stderr.write("Error creating regex object from string: " + s + "\n")
            return None
        reg_str = vals[0]
        reg_key = vals[1]
        reg_rep_str = vals[2] if vals[2] != "null" else None
        reg_store = vals[3] if vals[3] != "null" and len(vals[3]) > 0 else None

        return regex_tuple(reg_str,[reg_key],reg_rep_str, reg_store)

class json_streamer(object):
    def __init__(self, output_keys, regex_tuple_list, json_indent=JSON_INDENT):
        self.output_keys = output_keys
        self.regex_tuple_list = regex_tuple_list
        self.json_indent = json_indent
        self.regex_dict = None
        self.create_regex_dict()
    
    def create_regex_dict(self):
        d = {}
        for item in self.regex_tuple_list:
            for k in item.keys_to_process:
                if k in d:
                    d[k].append(item)
                else:
                    d[k] = [item]
        self.regex_dict = d
    
    def process_json_object(self, stream_dict):
        # for every registered key, look whether we have one in our stream object
        for key, r_list in self.regex_dict.items():
            if key in stream_dict:
                for r in r_list:
                    if r.key_to_store_match is not None:
                        # looking for all occurences and storing them in an json key
                        matches = r.regexp_obj.findall(stream_dict[key])
                        stream_dict[r.key_to_store_match] = matches # TODO: can not handle multiple rules storing in the same key!!!
                    if r.replace_str is not None:
                        # replacing all occurences
                        stream_dict[key] = r.regexp_obj.sub(r.replace_str, stream_dict[key])
    
    def main_stream(self, stream_input = sys.stdin, stream_output = sys.stdout, stream_error = sys.stderr):
        processed_buffer= ""
        depth = 0               # bracket depth (in case we have to handle nested json objects)
        inside_quotes = False   # used for deteting whether we are inside a string in order to ignore brackets inside quotes
        line_counter = 0
        current_batch_start_line = 0
        success_batch_counter = 0
        fail_batch_counter = 0

        prev_c = ''

        for line in stream_input:
            line_counter+=1
            for c in line:
                if c == '{' and not inside_quotes:
                    if depth == 0:
                        current_batch_start_line = line_counter
                    depth += 1
                    processed_buffer += c
                elif c == '}' and depth > 0 and not inside_quotes:
                    depth -= 1
                    processed_buffer += c
                    if depth == 0:
                        try:
                            d = json.loads(processed_buffer)
                            self.process_json_object(d)
                            stream_output.write(json.dumps(d, indent=self.json_indent, ensure_ascii=False))
                            processed_buffer = ""
                            success_batch_counter += 1
                        except json.decoder.JSONDecodeError:
                            stream_error.write("Error processing json object. Ignoring the following lines (starting at line " 
                                + str(current_batch_start_line) + "):\n\n")
                            stream_error.write(processed_buffer + "\n\n")
                            processed_buffer = ""
                            fail_batch_counter += 1

                elif depth > 0:
                    processed_buffer += c
                else:
                    stream_output.write(c)

                # flipping quotes status (and don't forget to exlude escaped quotes!)
                if c == '"' and prev_c != '\\':
                    inside_quotes = not inside_quotes
                
                # setting previous c. only exception: if a double backslash (= escaped backslash) occurs ignore it.
                # Because that would break our escaped character detection
                prev_c = c if not (c == '\\' and prev_c == '\\') else ''
        
        stream_error.write("\n\nReached EOF. #Processed objects: " + str(success_batch_counter) + ", #failed objects: " + str(fail_batch_counter) + "\n\n")

if __name__ == "__main__":

    args = sys.argv[1:]

    if len(args) == 0:
        print("missing arguments")
        sys.exit(-1)
    
    reg_tuples = []
    while len(args) > 0:
        reg_tuples.append(regex_tuple.create_from_string(args[0]))
        args = args[1:]
    
    streamer = json_streamer(None, reg_tuples)
    streamer.main_stream()

