--[[

DOCUMENTATION:

the "tableToInput" value is the table for the input.

the "mode" value is either for a circle or a square, 1 expected output and 0 for wrong answer.

the "debug" value shows the debug information like the bias and the table values, if its 0 it will just print "1" or "0".
(must be a 1 or a 0)

if the "tryagain" value is 1 then it will try again without exiting the function and if it is 0 then it will return and
exit the function even if it is not the expected output
]]--

local function main(tableToInput,mode,debug,tryagain)

    --creating the tables
    local currentvalues = {}
    local inputweights = {}

    repeat
        table.insert(currentvalues,0)
    until #currentvalues == #tableToInput

    repeat
        table.insert(inputweights,0)
    until #inputweights == #tableToInput

    --actual neural network stuff
    local valueToReturn = 0
    local output = 0
    local outputBias = 3

    ::startofnet::

    local current = 1

    while current < #tableToInput do
        currentvalues[current] = tableToInput[current]*inputweights[current]
        current = current + 1
    end

    current = 1
    local currentadd = 0

    while current < #tableToInput do
        currentadd = currentadd + currentvalues[current]
        current = current + 1
    end

    if currentadd > outputBias then
        if mode == 1 then
            if debug == 1 then
                print("1\n\ncurrentadd: "..currentadd.."\noutput bias: "..outputBias.."\n\nTABLE VALUES:\n\nTHESE ARENT DONE YET")
            else
                print("1")
            end
            valueToReturn = 1
            return valueToReturn
        else
            if debug == 1 then
                print("0\n\ncurrentadd: "..currentadd.."\noutput bias: "..outputBias.."\n\nTABLE VALUES:\n\nTHESE ARENT DONE YET")
            else
                print("0")
            end
            valueToReturn = 0
            --there's gotta be a better way to do this
        end
    elseif currentadd < outputBias then
        if mode == 0 then
            if debug == 1 then
                print("1\n\ncurrentadd: "..currentadd.."\noutput bias: "..outputBias.."\n\nTABLE VALUES:\n\nTHESE ARENT DONE YET")
            else
                print("1")
            end
            valueToReturn = 1
            return valueToReturn
        else
            if debug == 1 then
                print("0\n\ncurrentadd: "..currentadd.."\noutput bias: "..outputBias.."\n\nTABLE VALUES:\n\nTHESE ARENT DONE YET")
            else
                print("0")
            end
            valueToReturn = 0
            --there's gotta be a better way to do this
        end
    end

    --didnt fire
    current = 1
    if mode == 1 then
        while current < #tableToInput do
            inputweights[current] = inputweights[current] + tableToInput[current]
            current = current + 1
        end
    elseif mode == 0 then
        while current < #tableToInput do
            inputweights[current] = inputweights[current] - tableToInput[current]
            current = current + 1
        end
    end

    current = 1

    if tryagain then
        goto startofnet
    else
        return valueToReturn
    end
end

local wasd = {
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
}

local thisisavalue = 0

thisisavalue = main(wasd,0,1,0)
