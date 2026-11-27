classdef Client < handle
    %CLIENT Byte-oriented adapter to the OceanStorage Rust child process.
    % Missing values return found=false; an empty stored value has found=true.
    properties (Access = private)
        Process
        Reader
        Writer
    end
    methods
        function obj = Client(executable, directory)
            arguments
                executable (1,1) string
                directory (1,1) string
            end
            commands = java.util.ArrayList();
            commands.add(java.lang.String(executable));
            commands.add(java.lang.String('serve'));
            commands.add(java.lang.String(directory));
            builder = java.lang.ProcessBuilder(commands);
            obj.Process = builder.start();
            obj.Reader = java.io.BufferedReader(java.io.InputStreamReader(obj.Process.getInputStream(), 'US-ASCII'));
            obj.Writer = java.io.BufferedWriter(java.io.OutputStreamWriter(obj.Process.getOutputStream(), 'US-ASCII'));
            ready = obj.Reader.readLine();
            if isempty(ready) || string(ready) ~= "READY" + char(9) + "1"
                obj.Process.destroy();
                error('OceanStorage:OpenFailed', 'Storage engine did not start. Check path and directory ownership.');
            end
        end
        function put(obj, key, value)
            arguments
                obj (1,1) OceanStorage.Client
                key (1,:) uint8
                value (1,:) uint8
            end
            obj.expectOK(obj.request("PUT", obj.hex(key), obj.hex(value)));
        end
        function [value, found] = get(obj, key)
            arguments
                obj (1,1) OceanStorage.Client
                key (1,:) uint8
            end
            fields = obj.request("GET", obj.hex(key));
            found = fields(1) == "VALUE";
            value = uint8.empty(1,0);
            if found
                value = obj.unhex(fields(2));
            elseif fields(1) ~= "MISSING"
                error('OceanStorage:Protocol', 'Unexpected response.');
            end
        end
        function remove(obj, key)
            arguments
                obj (1,1) OceanStorage.Client
                key (1,:) uint8
            end
            obj.expectOK(obj.request("DELETE", obj.hex(key)));
        end
        function rows = scan(obj, startKey, endKey)
            arguments
                obj (1,1) OceanStorage.Client
                startKey (1,:) uint8 = uint8.empty(1,0)
                endKey = []
            end
            endpoint = "*";
            if ~isequal(endKey, [])
                validateattributes(endKey, {'uint8'}, {'row'});
                endpoint = obj.hex(endKey);
            end
            fields = obj.request("SCAN", obj.hex(startKey), endpoint);
            % Linked list avoids repeatedly resizing the result cell array.
            head = []; count = 0;
            while fields(1) ~= "END"
                if numel(fields) ~= 3 || fields(1) ~= "ROW"
                    error('OceanStorage:Protocol', 'Unexpected scan response.');
                end
                head = struct('Key', obj.unhex(fields(2)), 'Value', obj.unhex(fields(3)), 'Previous', head);
                count = count + 1;
                fields = obj.read();
            end
            rows = cell(count,2);
            for index = count:-1:1
                rows(index,:) = {head.Key, head.Value};
                head = head.Previous;
            end
        end
        function result = inspect(obj)
            arguments
                obj (1,1) OceanStorage.Client
            end
            fields = obj.request("INSPECT");
            if fields(1) ~= "TEXT"
                error('OceanStorage:Protocol', 'Unexpected response.');
            end
            result = string(native2unicode(obj.unhex(fields(2)), 'UTF-8'));
        end
        function close(obj)
            arguments
                obj (1,1) OceanStorage.Client
            end
            if ~isempty(obj.Process)
                cleanup = onCleanup(@() obj.terminate());
                if obj.Process.isAlive()
                    obj.expectOK(obj.request("CLOSE"));
                end
            end
        end
        function delete(obj)
            arguments
                obj
            end
            for index = 1:numel(obj)
                obj(index).terminate();
            end
        end
    end
    methods (Access = private)
        function fields = request(obj, varargin)
            if isempty(obj.Process) || ~obj.Process.isAlive()
                error('OceanStorage:Closed', 'Storage engine is closed.');
            end
            obj.Writer.write(char(strjoin([varargin{:}], char(9))));
            obj.Writer.newLine(); obj.Writer.flush();
            fields = obj.read();
        end
        function fields = read(obj)
            line = obj.Reader.readLine();
            if isempty(line)
                error('OceanStorage:Disconnected', 'Connection ended; last write may have committed.');
            end
            fields = split(string(line), char(9));
            if fields(1) == "ERR"
                error('OceanStorage:Engine', '%s', native2unicode(obj.unhex(fields(2)), 'UTF-8'));
            end
        end
        function terminate(obj)
            if ~isempty(obj.Process)
                obj.Process.destroy();
                obj.Process.waitFor();
                obj.Reader.close(); obj.Writer.close();
                obj.Process = [];
            end
        end
    end
    methods (Static, Access = private)
        function text = hex(bytes)
            text = string(reshape(dec2hex(bytes,2).',1,[]));
        end
        function bytes = unhex(text)
            if strlength(text) == 0
                bytes = uint8.empty(1,0);
            else
                bytes = reshape(uint8(hex2dec(reshape(char(text),2,[]).')),1,[]);
            end
        end
        function expectOK(fields)
            if ~isequal(fields, "OK")
                error('OceanStorage:Protocol', 'Unexpected response.');
            end
        end
    end
end
