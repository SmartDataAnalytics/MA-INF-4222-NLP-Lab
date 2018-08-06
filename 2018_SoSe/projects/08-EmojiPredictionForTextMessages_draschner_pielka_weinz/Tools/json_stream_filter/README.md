# JSON stream filter

a little tool for performing regex operations on string-values in json files (or streams) 

----

## Basics

let this be an example set:

```json
{
  "id": "obj_1",
  "key1": "some example text! 1234",
  "key2": "another example"
}
{
  "id": "obj_2",
  "key1": "...",
  "key2": "..."
}
```

an example regex operation replacing all numbers in all `key1`-fields with the string `NUMBER` and storing all matches in a new field called `found_numbers` can be defined as:

```
;[0-9]+;key1;NUMBER;found_numbers
```

the first character (here `;`) is used as separater for the following fields. It can be any unicode character. The following fields in detail:

* `[0-9]+`: the regex expression which is used for finding matches. In this case at least one character in the range 0-9
* `key1`: the key on which the expression is performed. All other keys are ignored
* `NUMBER` the string to replace found matches. Can be any string (also an empty string) or regular expression (which is accepted by pythons `re` library for a substitution).
  * **NOTE**: use `null` here for not doing any substitution
* `found_numbers`: key name for storing found matches as a list there. Set to an empty string for not storing the matches

the result for the example:

```json
{
  "id": "obj_1",
  "key1": "some example text! NUMBER",
  "key2": "another example",
  "found_numbers": [
    "1234"
  ]
}
{
  "id": "obj_2",
  "key1": "...",
  "key2": "...",
  "found_numbers": []
}  
```

----

## Command line interface

just run the python file with every regex operation as own argument. Output is written to stdout, progress information and errors to stderr. Input will be collected by stdin.

Assuming the example file above is stored in `example.json` and we can store the result to `result.json` by doing

```bash
cat example.json | json_stream_filter.py ";[0-9]+;key1;NUMBER;found_numbers" > result.json
```

----

## using as a python module

the file can regularly be imported.  It contains two classes:

* `regex_tuple`: for handling the regex operations (like mentioned above). A tuple can be initialized either as a string like above:

  * `regex_tuple.create_from_string(s)`

    or by passing the already separated fields to the constructor:
  
  * `regex_tuple(regexp_str, keys_to_process, replace_str = None, key_to_store_match = None)`
      * **NOTE**: `keys_to_process` has to be a list of keys (for future modifications), but so far only one key is supported, so at the moment this should be initialized as a list of one key

the second class is the file processor:

* `json_streamer`: is initialized the following way:

  * `json_streamer(output_keys, regex_tuple_list, json_indent=JSON_INDENT)`
    * `output_keys`: ignored so far. will be used for just filtering out by key names
    * `regex_tuple_list` list of used `regex_tuple` objects
    *  `json_indent` indent of formatted json output. Set to `2` by default

  for starting the main filter process:

  * `main_stream(stream_input=sys.stdin, stream_output=sys.stdout, stream_error=sys.stderr)`
    * by default it is using the default system inputs/outputs, but any file object can be set as input/output parameter