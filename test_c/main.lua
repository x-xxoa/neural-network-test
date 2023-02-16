--[[
    
    ::::DOCUMENTATION::::

the intable is the input for the neural network, please know that this neural network uses tables a lot
so to interface with it you need to use tables.

the layercount is the amount of hidden layers for the neural network, the amount of nodes in each layer
is calculated from a 2 point slope, (y2-y1)/(x1-x2), where y2 is the output node count, y1 is the input
node count, x1 is 0 and x2 is the layercount, the amount of hidden layers in the neural network.

the outcount is the total amount of output nodes, even if it is 1 it will output a table.

feel free to use and improve this as long as you give credit, i'm making this open source because i want
people who want to learn about neural networks and ai to be able to understand it easier and be able to
create their own.

thats all, and remember this is the first release so it might be a bit buggy.

oh and also more information is below and a copy of this is in the README.txt file. the reason i created
this is because i don't know how to use luarocks (and also because i was bored and wanted to learn about
neural networks)

yt: @xxoa_
github: x-xxoa

    ::::MORE INFORMATION::::

i'll do some updates to this over time and once it's all done and i've confirmed everything works i'll create
a seperate github repository, include support for multiple neural networks in 1 file and create more functions.

some of the stuff i'll work on once i've confirmed the back-propagation algorithm works is attempt to catch
errors before they happen, create an initialize function with support for multiple neural networks in 1 file
using, id system and fix some other stuff and also optimize the neural network and the back-propagation
algorithm.

    ::::STRENGTHS & LIMITATIONS::::

S: because this neural network uses Lua it can be faster than some neural networks made in python, might
not be as fast as some but because Lua is faster than python it could be faster than some. also because
i'm using Lua it is readable and easy to understand.

S: because the things are enclosed in functions it is easy to implement into code, all you need to do is
paste the functions or import the functions

S: because this uses no external libraries that means that all this requires is bare Lua 5.4 to run (might
not work on older versions of Lua.) and no package manager which is good for security of the thing because
you can read the code and decide if it's safe, if you don't know how to install and manage packages and if
a package gets updated and some feature gets depracated or broken that wouldn't happen (unless new versions
of Lua break the code)

S: because it creates the tables in _G the training progress is saved throughout the entire file and all you
have to do to train is do adjust(parameter,parameter,paramerter) and main(parameter,parameter,parameter)

L: because this uses the sigmoid activation function it might not be the best for all use cases. i've tried
using other activation functions but they didnt work and the sigmoid function worked. (ill make an option for
the tanh function in the next, improved, library version.)

L: this is still part of a test, so i wouldn't reccommend using this in actual serious situations.

L: because it uses _G it could clutter variables and could be slower but currently i don't think there's
a way besides just putting the premade tables into the functions.

L: because of the way this works you can only have 1 neural network per file, the intable can change but changing
layercount and outcount could lead to an error or incorrect results.

    ::::HOW IT WORKS::::

the neural networks starts by creating the functions, the first and only function created is getlayer(). getlayer()
takes the last layer, the next layer, the weight layer inbetween and the bias table for the next layer. it then
declares 1 variable, sum, and loops for the amount of items in the nextlayer (#nextlayer, loop variable is a).
then in that loop thereis another loop that loops for the amount of items in the lastlayer (#lastlayer, loop
variable is i), then it adds the i'th item in lastlayer multiplied by the i+(a-1) multiplied by the amount of items
in the last layer (#lastlayer). then the a'th item in the next layer (nextlayer) is set to sum + the current node's
bias (biases[a]) put into a sigmoid function.

the input is fed into the neural network in the form of a table, this is because it's easy to work with tables and if
you want to read an image like BMP or PNG you would have to write code to convert it to a table. it then checks if
_G["stc"] is nil and if it is set it to false, then if _G["stc"] is false it will create the current and bias tables
for the hidden layers, then the weight tables for the hidden layers, then the output current and bias tables and
finally the output weight tables. it then sets _G["stc"] to true. if _G["stc"] is true it will skip over the table
creation. the created tables are stored in the global environment (_G) to be accessed throughout the entire file.

then getlayer() is called for the input and the first hidden layer and then a loop for the amount of hidden layers
starting at 2 (loop variable is i) calls getlayer() for the last layer (_G["c"..i-1]) and the next layer
(_G["c"..i]). once the loop is done it calles getlayer() again for the lastlayer in the hidden layers and the output
nodes. it then returns the output (_G["o"]).

the adjust function starts by checking if out and expected out are the same size to prevent an error, it then gets
the weighted sum of the weights and puts it into the derivative of the sigmoid function. before it does anything
else it checks if the averaged mse for the output is below a certain value to avoid overfitting and if it is then
return true. it then calculates the gradient descent table for the weights and then the gradient descent table for
the biases, it adjusts the output layer weights based on gradw and then the rest of the weights, also based on gradw.
after that it adjusts the output biases based on gradb, and then adjusts the rest of the biases, also based on gradb.

    ::::POTENTIAL OPTIMIZATIONS::::

some potential optimizations i know at the top of my head while writing this, which is in school during lunch at
11:05 am on thursday, febuary 16th, 2023, are all for the back-propagation algorithm. there are optimizations that are
possible for the forward-pass algorithm but i'm not able to come up with any right now. after the output weights
are adjusted it then adjusts the rest of the weights, then again on the next gradw item, then again and so on.
this can be optimized by subtracting all of them together and then subtracting the weights from that, same with
the biases.

another potential optimization is having a dataset parameter for the adjust() function. the dataset parameter
could be a table with a list of inputs, outputs and expected outputs.

    ::::HOW TO USE THE FUNCTIONS::::

hopefully from the above documentation you've figured out how to use the code. if not, there is some example code
below the functions. if you still don't get it from the example code i'll explain.

the main() function takes a table for the first parameter, a number for the second parameter and a number for the
last and third parameter, intable, layercount and outcount respectively. the intable is the input for the neural
network, the layercount is the amount of hidden layers for the neural network, and the outcount is the amount of
output nodes for the neural network. the main() function returns a table as the output, even if the outcount is
1 it still returns a table.

the adjust() function takes a table for the first parameter, a table for the second and third parameter and a
number for the fourth parameter, intable, out, expectedout and learningrate respectively. the intable is the
input for the neural network you want to adjust for, the out is the real output, the expectedout is the expected
output (the desired output for that input) and the learning rate is usually a low number like 0.01 or 0.001. the
adjust() function returns true if the mse is low enough so it's accurate enough (to avoid overfitting), and
returns false if it had to adjust the weights and biases. so for an example we could have code like this:

--CODE--

local tabletoinput = {0,1,1,1}
local desiredoutput = {1,0,0,0}

local realoutput = main(tabletoinput,2,4)

repeat
    realoutput = main(tabletoinput,2,4)
until adjust(tabletoinput,realoutput,desiredoutput,0.01)

--CODE--

thats the end of the documentation, it's currently 11:22 am on thursday, febuary 16th, 2023.

]]--

local function main(intable,layercount,outcount)
    --declare the functions
    local function getlayer(lastlayer,nextlayer,weights,biases)
        --declare the variables
        local sum = 0

        --get the sum of the connected weights to the current node we are on and replace the nextlayer
        for a = 1,#nextlayer do
            for i = 1,#lastlayer do
                sum = sum + lastlayer[i]*(weights[i+((a-1)*#lastlayer)])
            end
            nextlayer[a] = 1 / (1 + math.exp(-sum+biases[a]))
            sum = 0
        end
    end

    --declare the variables
    if _G["stc"] == nil then --skiptablecreation
        _G["stc"] = false
    else
        _G["stc"] = true
    end
    _G["oc"] = outcount --outcount (for transferring to training function)
    _G["lc"] = layercount --layercount (for transferring to training function)
    local tablereturn = 0
    local amounttofill = 0

    --create the tables
    if not _G["stc"] then
        --create the tables for the node values (bias and current)
        for i = 1,layercount do
            local ctablename = "c"..i
            local btablename = "b"..i
            _G[ctablename] = {}
            _G[btablename] = {}
            
            amounttofill = math.ceil(((outcount - #intable) * i / (layercount - 0)) + #intable)

            for a = 1,amounttofill do
                _G[ctablename][a] = 0.0
                _G[btablename][a] = 0.0
            end
        end

        --create the tables for the connection values (weight)
        for i = 1,layercount + 1 do
            local wtablename = "w"..i
            _G[wtablename] = {}
            
            if i > 1 then --get the amount to fill
                amounttofill = math.ceil(((outcount - #intable) * (i - 1) / (layercount - 0)) + #intable)*math.ceil(((outcount - #intable) * i / (layercount - 0)) + #intable)
            else
                amounttofill = #intable*math.ceil(((outcount - #intable) * i / (layercount - 0)) + #intable)
            end

            for a = 1,amounttofill do
                _G[wtablename][a] = 0.0
            end
        end
        
        --create the tables for the output (bias and current)
        _G["ob"] = {}
        _G["o"] = {}
        for i = 1,outcount do
            _G["ob"][i] = 0.0
            _G["o"][i] = 0.0
        end

        --create the tables for the output connection (weight)
        _G["ow"] = {}
        for i = 1,outcount*math.ceil(((outcount - #intable) * layercount / (layercount - 0)) + #intable) do
            _G["ow"][i] = 0.0
        end
        
        _G["stc"] = true
    end

    --do the stuff
    getlayer(intable,_G["c1"],_G["w1"],_G["b1"]) --input layer to first hidden
    for i = 2,layercount,1 do --rest of the hidden layers
        getlayer(_G["c"..i-1],_G["c"..i],_G["w"..i],_G["b"..i])
    end
    getlayer(_G["c"..layercount],_G["o"],_G["ow"],_G["ob"])

    return _G["o"]
end

--TRAINING!!!!!!!!!!! YEEAAAHHHHHH (i dont think making comments like this will be good for a job in the future)

local function adjust(intable,out,expectedout,learningrate,cross)
    --check if out and expectedout are the same size
    if #out ~= #expectedout then
        print("out and expectedout are not the same size, exiting.")
        os.exit()
    end

    --declare the functions
    local function calcmse(output,expectedoutput) --function for calculating the mean squared error. this is used to calculate the error between the the expected output and the real output, lower values are better but don't overshoot!
        local mse = 0
        for i = 1, #output do
            local error = output[i] - expectedoutput[i]
            mse = mse + error * error
        end
        return mse / #output
    end

    local function sigmoid(x,derivative)
        if derivative then
            return (1 / (1 + math.exp(-x))) * (1-(1 / (1 + math.exp(-x))))
        else
            return 1 / (1 + math.exp(-x))
        end
    end

    --declare the variables
    local mse = calcmse(out,expectedout)
    local costtable = {}
    local acost = 0
    local gradw = {}
    local gradb = {}
    local weightedsum = 0
    local dsig_wsum = 0

    --get the sum of the weighted inputs
    for a = 1,#_G["c1"] do
        for i = 1,#intable do
            weightedsum = weightedsum + _G["w1"][i+((a-1)*#intable)]*intable[i]
        end
    end

    --get dsig_wsum
    dsig_wsum = sigmoid(weightedsum,true)

    --get gradw
    for i = 1,#out do
        gradw[i] = ((out[i]-expectedout[i])^2*dsig_wsum)*learningrate
    end

    --get gradb
    for i = 1,#out do
        gradb[i] = ((out[i]-expectedout[i])*dsig_wsum)*learningrate
    end
    
    --adjust weights

    --adjust the output layer weights
    for a = 1,#out do
        for i = 1,#_G["c".._G["lc"]] do
            _G["ow"][i+((a-1)*#_G["c".._G["lc"]])] = _G["ow"][i+((a-1)*#_G["c".._G["lc"]])] - gradw[i]
        end
    end

    --adjust the rest of the weights
    for a = 1,#out do
        for b = _G["lc"],1,-1 do
            for i = 1,#_G["w"..b] do
                _G["w"..b][i] = _G["w"..b][i] - gradw[a]
            end
        end
    end
    
    --adjust biases

    --adjust the output layer biases
    for i = 1,#out do
        _G["ob"][i] = _G["ob"][i] - gradb[i]
    end

    --adjust the rest of the biases
    for a = 1,#out do
        for b = _G["lc"],1,-1 do
            for i = 1,#_G["b"..b] do
                _G["b"..b][i] = _G["b"..b][i] - gradb[a]
            end
        end
    end
end

local a = {
    0,1,1,1
}

local eo = {
    1,0,0,0
}

local b = main(a,2,4)

for i,v in pairs(b) do
    print(i,v)
end

for i = 1,2000 do
    if adjust(a,b,eo,0.01,false) then break end
    local b = main(a,2,4)

    for i,v in pairs(b) do
        print(i,v)
    end
end
