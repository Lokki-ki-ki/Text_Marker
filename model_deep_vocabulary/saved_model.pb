Í(
Ţ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.9.22v2.9.1-132-g18960c44ad38Ź$

Adam/sequential_2/Output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/sequential_2/Output/bias/v

3Adam/sequential_2/Output/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential_2/Output/bias/v*
_output_shapes
:*
dtype0

!Adam/sequential_2/Output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/sequential_2/Output/kernel/v

5Adam/sequential_2/Output/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_2/Output/kernel/v*
_output_shapes

:*
dtype0

 Adam/sequential_2/Hidden1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/sequential_2/Hidden1/bias/v

4Adam/sequential_2/Hidden1/bias/v/Read/ReadVariableOpReadVariableOp Adam/sequential_2/Hidden1/bias/v*
_output_shapes
:*
dtype0
 
"Adam/sequential_2/Hidden1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"Adam/sequential_2/Hidden1/kernel/v

6Adam/sequential_2/Hidden1/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/sequential_2/Hidden1/kernel/v*
_output_shapes

:*
dtype0

 Adam/sequential_2/Hidden0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/sequential_2/Hidden0/bias/v

4Adam/sequential_2/Hidden0/bias/v/Read/ReadVariableOpReadVariableOp Adam/sequential_2/Hidden0/bias/v*
_output_shapes
:*
dtype0
 
"Adam/sequential_2/Hidden0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*3
shared_name$"Adam/sequential_2/Hidden0/kernel/v

6Adam/sequential_2/Hidden0/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/sequential_2/Hidden0/kernel/v*
_output_shapes

:#*
dtype0

Adam/sequential_2/Output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/sequential_2/Output/bias/m

3Adam/sequential_2/Output/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential_2/Output/bias/m*
_output_shapes
:*
dtype0

!Adam/sequential_2/Output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/sequential_2/Output/kernel/m

5Adam/sequential_2/Output/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_2/Output/kernel/m*
_output_shapes

:*
dtype0

 Adam/sequential_2/Hidden1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/sequential_2/Hidden1/bias/m

4Adam/sequential_2/Hidden1/bias/m/Read/ReadVariableOpReadVariableOp Adam/sequential_2/Hidden1/bias/m*
_output_shapes
:*
dtype0
 
"Adam/sequential_2/Hidden1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"Adam/sequential_2/Hidden1/kernel/m

6Adam/sequential_2/Hidden1/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/sequential_2/Hidden1/kernel/m*
_output_shapes

:*
dtype0

 Adam/sequential_2/Hidden0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/sequential_2/Hidden0/bias/m

4Adam/sequential_2/Hidden0/bias/m/Read/ReadVariableOpReadVariableOp Adam/sequential_2/Hidden0/bias/m*
_output_shapes
:*
dtype0
 
"Adam/sequential_2/Hidden0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*3
shared_name$"Adam/sequential_2/Hidden0/kernel/m

6Adam/sequential_2/Hidden0/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/sequential_2/Hidden0/kernel/m*
_output_shapes

:#*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

sequential_2/Output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namesequential_2/Output/bias

,sequential_2/Output/bias/Read/ReadVariableOpReadVariableOpsequential_2/Output/bias*
_output_shapes
:*
dtype0

sequential_2/Output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_namesequential_2/Output/kernel

.sequential_2/Output/kernel/Read/ReadVariableOpReadVariableOpsequential_2/Output/kernel*
_output_shapes

:*
dtype0

sequential_2/Hidden1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namesequential_2/Hidden1/bias

-sequential_2/Hidden1/bias/Read/ReadVariableOpReadVariableOpsequential_2/Hidden1/bias*
_output_shapes
:*
dtype0

sequential_2/Hidden1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_namesequential_2/Hidden1/kernel

/sequential_2/Hidden1/kernel/Read/ReadVariableOpReadVariableOpsequential_2/Hidden1/kernel*
_output_shapes

:*
dtype0

sequential_2/Hidden0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namesequential_2/Hidden0/bias

-sequential_2/Hidden0/bias/Read/ReadVariableOpReadVariableOpsequential_2/Hidden0/bias*
_output_shapes
:*
dtype0

sequential_2/Hidden0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*,
shared_namesequential_2/Hidden0/kernel

/sequential_2/Hidden0/kernel/Read/ReadVariableOpReadVariableOpsequential_2/Hidden0/kernel*
_output_shapes

:#*
dtype0

NoOpNoOp
ů7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*´7
valueŞ7B§7 B 7
ć
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
_build_input_shape

signatures*
´
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_feature_columns

_resources* 
Ś
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
Ś
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias*
Ś
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias*
.
0
1
%2
&3
-4
.5*
.
0
1
%2
&3
-4
.5*
* 
°
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
4trace_0
5trace_1
6trace_2
7trace_3* 
6
8trace_0
9trace_1
:trace_2
;trace_3* 
* 
°
<iter

=beta_1

>beta_2
	?decay
@learning_ratemqmr%ms&mt-mu.mvvwvx%vy&vz-v{.v|*
* 

Aserving_default* 
* 
* 
* 

Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Gtrace_0
Htrace_1* 

Itrace_0
Jtrace_1* 
* 
* 

0
1*

0
1*
* 

Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ptrace_0* 

Qtrace_0* 
ke
VARIABLE_VALUEsequential_2/Hidden0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEsequential_2/Hidden0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

%0
&1*

%0
&1*
* 

Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

Wtrace_0* 

Xtrace_0* 
ke
VARIABLE_VALUEsequential_2/Hidden1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEsequential_2/Hidden1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

-0
.1*

-0
.1*
* 

Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

^trace_0* 

_trace_0* 
jd
VARIABLE_VALUEsequential_2/Output/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEsequential_2/Output/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

`0
a1
b2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
c	variables
d	keras_api
	etotal
	fcount*
H
g	variables
h	keras_api
	itotal
	jcount
k
_fn_kwargs*
H
l	variables
m	keras_api
	ntotal
	ocount
p
_fn_kwargs*

e0
f1*

c	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

i0
j1*

g	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

n0
o1*

l	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

VARIABLE_VALUE"Adam/sequential_2/Hidden0/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/sequential_2/Hidden0/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/sequential_2/Hidden1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/sequential_2/Hidden1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/sequential_2/Output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/sequential_2/Output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/sequential_2/Hidden0/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/sequential_2/Hidden0/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/sequential_2/Hidden1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE Adam/sequential_2/Hidden1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE!Adam/sequential_2/Output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/sequential_2/Output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
n
serving_default_ARIPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙

$serving_default_Incorrect_form_ratioPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_av_word_per_senPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_coherence_scorePlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
s
serving_default_cohesionPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
y
serving_default_corrected_textPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

,serving_default_dale_chall_readability_scorePlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

$serving_default_flesch_kincaid_gradePlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
~
#serving_default_flesch_reading_easePlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_freq_diff_wordsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_freq_of_adjPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_freq_of_advPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

$serving_default_freq_of_distinct_adjPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

$serving_default_freq_of_distinct_advPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_freq_of_nounPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_freq_of_pronounPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
}
"serving_default_freq_of_transitionPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_freq_of_verbPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
~
#serving_default_freq_of_wrong_wordsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

$serving_default_lexrank_avg_min_diffPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

%serving_default_lexrank_interquartilePlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
y
serving_default_mcalpine_eflawPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_noun_to_adjPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙

%serving_default_num_of_grammar_errorsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
}
"serving_default_num_of_short_formsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙

$serving_default_number_of_diff_wordsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
z
serving_default_number_of_wordsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0	*
shape:˙˙˙˙˙˙˙˙˙
{
 serving_default_phrase_diversityPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
w
serving_default_punctuationsPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
~
#serving_default_sentence_complexityPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
}
"serving_default_sentiment_compoundPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
}
"serving_default_sentiment_negativePlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
}
"serving_default_sentiment_positivePlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
~
#serving_default_stopwords_frequencyPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
x
serving_default_text_standardPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
n
serving_default_ttrPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
v
serving_default_verb_to_advPlaceholder*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
dtype0*
shape:˙˙˙˙˙˙˙˙˙
Î
StatefulPartitionedCallStatefulPartitionedCallserving_default_ARI$serving_default_Incorrect_form_ratioserving_default_av_word_per_senserving_default_coherence_scoreserving_default_cohesionserving_default_corrected_text,serving_default_dale_chall_readability_score$serving_default_flesch_kincaid_grade#serving_default_flesch_reading_easeserving_default_freq_diff_wordsserving_default_freq_of_adjserving_default_freq_of_adv$serving_default_freq_of_distinct_adj$serving_default_freq_of_distinct_advserving_default_freq_of_nounserving_default_freq_of_pronoun"serving_default_freq_of_transitionserving_default_freq_of_verb#serving_default_freq_of_wrong_words$serving_default_lexrank_avg_min_diff%serving_default_lexrank_interquartileserving_default_mcalpine_eflawserving_default_noun_to_adj%serving_default_num_of_grammar_errors"serving_default_num_of_short_forms$serving_default_number_of_diff_wordsserving_default_number_of_words serving_default_phrase_diversityserving_default_punctuations#serving_default_sentence_complexity"serving_default_sentiment_compound"serving_default_sentiment_negative"serving_default_sentiment_positive#serving_default_stopwords_frequencyserving_default_text_standardserving_default_ttrserving_default_verb_to_advsequential_2/Hidden0/kernelsequential_2/Hidden0/biassequential_2/Hidden1/kernelsequential_2/Hidden1/biassequential_2/Output/kernelsequential_2/Output/bias*6
Tin/
-2+					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

%&'()**-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_52452
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¸
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/sequential_2/Hidden0/kernel/Read/ReadVariableOp-sequential_2/Hidden0/bias/Read/ReadVariableOp/sequential_2/Hidden1/kernel/Read/ReadVariableOp-sequential_2/Hidden1/bias/Read/ReadVariableOp.sequential_2/Output/kernel/Read/ReadVariableOp,sequential_2/Output/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp6Adam/sequential_2/Hidden0/kernel/m/Read/ReadVariableOp4Adam/sequential_2/Hidden0/bias/m/Read/ReadVariableOp6Adam/sequential_2/Hidden1/kernel/m/Read/ReadVariableOp4Adam/sequential_2/Hidden1/bias/m/Read/ReadVariableOp5Adam/sequential_2/Output/kernel/m/Read/ReadVariableOp3Adam/sequential_2/Output/bias/m/Read/ReadVariableOp6Adam/sequential_2/Hidden0/kernel/v/Read/ReadVariableOp4Adam/sequential_2/Hidden0/bias/v/Read/ReadVariableOp6Adam/sequential_2/Hidden1/kernel/v/Read/ReadVariableOp4Adam/sequential_2/Hidden1/bias/v/Read/ReadVariableOp5Adam/sequential_2/Output/kernel/v/Read/ReadVariableOp3Adam/sequential_2/Output/bias/v/Read/ReadVariableOpConst**
Tin#
!2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_54473
ď
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential_2/Hidden0/kernelsequential_2/Hidden0/biassequential_2/Hidden1/kernelsequential_2/Hidden1/biassequential_2/Output/kernelsequential_2/Output/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_2count_2total_1count_1totalcount"Adam/sequential_2/Hidden0/kernel/m Adam/sequential_2/Hidden0/bias/m"Adam/sequential_2/Hidden1/kernel/m Adam/sequential_2/Hidden1/bias/m!Adam/sequential_2/Output/kernel/mAdam/sequential_2/Output/bias/m"Adam/sequential_2/Hidden0/kernel/v Adam/sequential_2/Hidden0/bias/v"Adam/sequential_2/Hidden1/kernel/v Adam/sequential_2/Hidden1/bias/v!Adam/sequential_2/Output/kernel/vAdam/sequential_2/Output/bias/v*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_54570Ó"
ź

&__inference_Output_layer_call_fn_54317

inputs
unknown:
	unknown_0:
identity˘StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_51532o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
á

 __inference__wrapped_model_51010
ari	
incorrect_form_ratio
av_word_per_sen
coherence_score
cohesion
corrected_text 
dale_chall_readability_score
flesch_kincaid_grade
flesch_reading_ease
freq_diff_words
freq_of_adj
freq_of_adv
freq_of_distinct_adj
freq_of_distinct_adv
freq_of_noun
freq_of_pronoun
freq_of_transition
freq_of_verb
freq_of_wrong_words
lexrank_avg_min_diff
lexrank_interquartile
mcalpine_eflaw
noun_to_adj
num_of_grammar_errors	
num_of_short_forms	
number_of_diff_words	
number_of_words	
phrase_diversity
punctuations
sentence_complexity
sentiment_compound
sentiment_negative
sentiment_positive
stopwords_frequency
text_standard
ttr
verb_to_advE
3sequential_2_hidden0_matmul_readvariableop_resource:#B
4sequential_2_hidden0_biasadd_readvariableop_resource:E
3sequential_2_hidden1_matmul_readvariableop_resource:B
4sequential_2_hidden1_biasadd_readvariableop_resource:D
2sequential_2_output_matmul_readvariableop_resource:A
3sequential_2_output_biasadd_readvariableop_resource:
identity˘+sequential_2/Hidden0/BiasAdd/ReadVariableOp˘*sequential_2/Hidden0/MatMul/ReadVariableOp˘+sequential_2/Hidden1/BiasAdd/ReadVariableOp˘*sequential_2/Hidden1/MatMul/ReadVariableOp˘*sequential_2/Output/BiasAdd/ReadVariableOp˘)sequential_2/Output/MatMul/ReadVariableOpy
.sequential_2/dense_features/ARI/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙¨
*sequential_2/dense_features/ARI/ExpandDims
ExpandDimsari7sequential_2/dense_features/ARI/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
$sequential_2/dense_features/ARI/CastCast3sequential_2/dense_features/ARI/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙}
%sequential_2/dense_features/ARI/ShapeShape(sequential_2/dense_features/ARI/Cast:y:0*
T0*
_output_shapes
:}
3sequential_2/dense_features/ARI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_2/dense_features/ARI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/dense_features/ARI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ń
-sequential_2/dense_features/ARI/strided_sliceStridedSlice.sequential_2/dense_features/ARI/Shape:output:0<sequential_2/dense_features/ARI/strided_slice/stack:output:0>sequential_2/dense_features/ARI/strided_slice/stack_1:output:0>sequential_2/dense_features/ARI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/sequential_2/dense_features/ARI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ő
-sequential_2/dense_features/ARI/Reshape/shapePack6sequential_2/dense_features/ARI/strided_slice:output:08sequential_2/dense_features/ARI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ć
'sequential_2/dense_features/ARI/ReshapeReshape(sequential_2/dense_features/ARI/Cast:y:06sequential_2/dense_features/ARI/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
?sequential_2/dense_features/Incorrect_form_ratio/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ű
;sequential_2/dense_features/Incorrect_form_ratio/ExpandDims
ExpandDimsincorrect_form_ratioHsequential_2/dense_features/Incorrect_form_ratio/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
6sequential_2/dense_features/Incorrect_form_ratio/ShapeShapeDsequential_2/dense_features/Incorrect_form_ratio/ExpandDims:output:0*
T0*
_output_shapes
:
Dsequential_2/dense_features/Incorrect_form_ratio/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Fsequential_2/dense_features/Incorrect_form_ratio/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Fsequential_2/dense_features/Incorrect_form_ratio/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ć
>sequential_2/dense_features/Incorrect_form_ratio/strided_sliceStridedSlice?sequential_2/dense_features/Incorrect_form_ratio/Shape:output:0Msequential_2/dense_features/Incorrect_form_ratio/strided_slice/stack:output:0Osequential_2/dense_features/Incorrect_form_ratio/strided_slice/stack_1:output:0Osequential_2/dense_features/Incorrect_form_ratio/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
@sequential_2/dense_features/Incorrect_form_ratio/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
>sequential_2/dense_features/Incorrect_form_ratio/Reshape/shapePackGsequential_2/dense_features/Incorrect_form_ratio/strided_slice:output:0Isequential_2/dense_features/Incorrect_form_ratio/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
8sequential_2/dense_features/Incorrect_form_ratio/ReshapeReshapeDsequential_2/dense_features/Incorrect_form_ratio/ExpandDims:output:0Gsequential_2/dense_features/Incorrect_form_ratio/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
:sequential_2/dense_features/av_word_per_sen/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ě
6sequential_2/dense_features/av_word_per_sen/ExpandDims
ExpandDimsav_word_per_senCsequential_2/dense_features/av_word_per_sen/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
1sequential_2/dense_features/av_word_per_sen/ShapeShape?sequential_2/dense_features/av_word_per_sen/ExpandDims:output:0*
T0*
_output_shapes
:
?sequential_2/dense_features/av_word_per_sen/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Asequential_2/dense_features/av_word_per_sen/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Asequential_2/dense_features/av_word_per_sen/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
9sequential_2/dense_features/av_word_per_sen/strided_sliceStridedSlice:sequential_2/dense_features/av_word_per_sen/Shape:output:0Hsequential_2/dense_features/av_word_per_sen/strided_slice/stack:output:0Jsequential_2/dense_features/av_word_per_sen/strided_slice/stack_1:output:0Jsequential_2/dense_features/av_word_per_sen/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;sequential_2/dense_features/av_word_per_sen/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ů
9sequential_2/dense_features/av_word_per_sen/Reshape/shapePackBsequential_2/dense_features/av_word_per_sen/strided_slice:output:0Dsequential_2/dense_features/av_word_per_sen/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ő
3sequential_2/dense_features/av_word_per_sen/ReshapeReshape?sequential_2/dense_features/av_word_per_sen/ExpandDims:output:0Bsequential_2/dense_features/av_word_per_sen/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
:sequential_2/dense_features/coherence_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ě
6sequential_2/dense_features/coherence_score/ExpandDims
ExpandDimscoherence_scoreCsequential_2/dense_features/coherence_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
1sequential_2/dense_features/coherence_score/ShapeShape?sequential_2/dense_features/coherence_score/ExpandDims:output:0*
T0*
_output_shapes
:
?sequential_2/dense_features/coherence_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Asequential_2/dense_features/coherence_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Asequential_2/dense_features/coherence_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
9sequential_2/dense_features/coherence_score/strided_sliceStridedSlice:sequential_2/dense_features/coherence_score/Shape:output:0Hsequential_2/dense_features/coherence_score/strided_slice/stack:output:0Jsequential_2/dense_features/coherence_score/strided_slice/stack_1:output:0Jsequential_2/dense_features/coherence_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;sequential_2/dense_features/coherence_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ů
9sequential_2/dense_features/coherence_score/Reshape/shapePackBsequential_2/dense_features/coherence_score/strided_slice:output:0Dsequential_2/dense_features/coherence_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ő
3sequential_2/dense_features/coherence_score/ReshapeReshape?sequential_2/dense_features/coherence_score/ExpandDims:output:0Bsequential_2/dense_features/coherence_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Gsequential_2/dense_features/dale_chall_readability_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ó
Csequential_2/dense_features/dale_chall_readability_score/ExpandDims
ExpandDimsdale_chall_readability_scorePsequential_2/dense_features/dale_chall_readability_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
>sequential_2/dense_features/dale_chall_readability_score/ShapeShapeLsequential_2/dense_features/dale_chall_readability_score/ExpandDims:output:0*
T0*
_output_shapes
:
Lsequential_2/dense_features/dale_chall_readability_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Nsequential_2/dense_features/dale_chall_readability_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Nsequential_2/dense_features/dale_chall_readability_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:î
Fsequential_2/dense_features/dale_chall_readability_score/strided_sliceStridedSliceGsequential_2/dense_features/dale_chall_readability_score/Shape:output:0Usequential_2/dense_features/dale_chall_readability_score/strided_slice/stack:output:0Wsequential_2/dense_features/dale_chall_readability_score/strided_slice/stack_1:output:0Wsequential_2/dense_features/dale_chall_readability_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Hsequential_2/dense_features/dale_chall_readability_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B : 
Fsequential_2/dense_features/dale_chall_readability_score/Reshape/shapePackOsequential_2/dense_features/dale_chall_readability_score/strided_slice:output:0Qsequential_2/dense_features/dale_chall_readability_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
@sequential_2/dense_features/dale_chall_readability_score/ReshapeReshapeLsequential_2/dense_features/dale_chall_readability_score/ExpandDims:output:0Osequential_2/dense_features/dale_chall_readability_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
?sequential_2/dense_features/flesch_kincaid_grade/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ű
;sequential_2/dense_features/flesch_kincaid_grade/ExpandDims
ExpandDimsflesch_kincaid_gradeHsequential_2/dense_features/flesch_kincaid_grade/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
6sequential_2/dense_features/flesch_kincaid_grade/ShapeShapeDsequential_2/dense_features/flesch_kincaid_grade/ExpandDims:output:0*
T0*
_output_shapes
:
Dsequential_2/dense_features/flesch_kincaid_grade/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Fsequential_2/dense_features/flesch_kincaid_grade/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Fsequential_2/dense_features/flesch_kincaid_grade/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ć
>sequential_2/dense_features/flesch_kincaid_grade/strided_sliceStridedSlice?sequential_2/dense_features/flesch_kincaid_grade/Shape:output:0Msequential_2/dense_features/flesch_kincaid_grade/strided_slice/stack:output:0Osequential_2/dense_features/flesch_kincaid_grade/strided_slice/stack_1:output:0Osequential_2/dense_features/flesch_kincaid_grade/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
@sequential_2/dense_features/flesch_kincaid_grade/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
>sequential_2/dense_features/flesch_kincaid_grade/Reshape/shapePackGsequential_2/dense_features/flesch_kincaid_grade/strided_slice:output:0Isequential_2/dense_features/flesch_kincaid_grade/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
8sequential_2/dense_features/flesch_kincaid_grade/ReshapeReshapeDsequential_2/dense_features/flesch_kincaid_grade/ExpandDims:output:0Gsequential_2/dense_features/flesch_kincaid_grade/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
>sequential_2/dense_features/flesch_reading_ease/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ř
:sequential_2/dense_features/flesch_reading_ease/ExpandDims
ExpandDimsflesch_reading_easeGsequential_2/dense_features/flesch_reading_ease/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
5sequential_2/dense_features/flesch_reading_ease/ShapeShapeCsequential_2/dense_features/flesch_reading_ease/ExpandDims:output:0*
T0*
_output_shapes
:
Csequential_2/dense_features/flesch_reading_ease/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Esequential_2/dense_features/flesch_reading_ease/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Esequential_2/dense_features/flesch_reading_ease/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Á
=sequential_2/dense_features/flesch_reading_ease/strided_sliceStridedSlice>sequential_2/dense_features/flesch_reading_ease/Shape:output:0Lsequential_2/dense_features/flesch_reading_ease/strided_slice/stack:output:0Nsequential_2/dense_features/flesch_reading_ease/strided_slice/stack_1:output:0Nsequential_2/dense_features/flesch_reading_ease/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?sequential_2/dense_features/flesch_reading_ease/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
=sequential_2/dense_features/flesch_reading_ease/Reshape/shapePackFsequential_2/dense_features/flesch_reading_ease/strided_slice:output:0Hsequential_2/dense_features/flesch_reading_ease/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
7sequential_2/dense_features/flesch_reading_ease/ReshapeReshapeCsequential_2/dense_features/flesch_reading_ease/ExpandDims:output:0Fsequential_2/dense_features/flesch_reading_ease/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
:sequential_2/dense_features/freq_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ě
6sequential_2/dense_features/freq_diff_words/ExpandDims
ExpandDimsfreq_diff_wordsCsequential_2/dense_features/freq_diff_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
1sequential_2/dense_features/freq_diff_words/ShapeShape?sequential_2/dense_features/freq_diff_words/ExpandDims:output:0*
T0*
_output_shapes
:
?sequential_2/dense_features/freq_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Asequential_2/dense_features/freq_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Asequential_2/dense_features/freq_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
9sequential_2/dense_features/freq_diff_words/strided_sliceStridedSlice:sequential_2/dense_features/freq_diff_words/Shape:output:0Hsequential_2/dense_features/freq_diff_words/strided_slice/stack:output:0Jsequential_2/dense_features/freq_diff_words/strided_slice/stack_1:output:0Jsequential_2/dense_features/freq_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;sequential_2/dense_features/freq_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ů
9sequential_2/dense_features/freq_diff_words/Reshape/shapePackBsequential_2/dense_features/freq_diff_words/strided_slice:output:0Dsequential_2/dense_features/freq_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ő
3sequential_2/dense_features/freq_diff_words/ReshapeReshape?sequential_2/dense_features/freq_diff_words/ExpandDims:output:0Bsequential_2/dense_features/freq_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
6sequential_2/dense_features/freq_of_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ŕ
2sequential_2/dense_features/freq_of_adj/ExpandDims
ExpandDimsfreq_of_adj?sequential_2/dense_features/freq_of_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
-sequential_2/dense_features/freq_of_adj/ShapeShape;sequential_2/dense_features/freq_of_adj/ExpandDims:output:0*
T0*
_output_shapes
:
;sequential_2/dense_features/freq_of_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=sequential_2/dense_features/freq_of_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=sequential_2/dense_features/freq_of_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/dense_features/freq_of_adj/strided_sliceStridedSlice6sequential_2/dense_features/freq_of_adj/Shape:output:0Dsequential_2/dense_features/freq_of_adj/strided_slice/stack:output:0Fsequential_2/dense_features/freq_of_adj/strided_slice/stack_1:output:0Fsequential_2/dense_features/freq_of_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
7sequential_2/dense_features/freq_of_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :í
5sequential_2/dense_features/freq_of_adj/Reshape/shapePack>sequential_2/dense_features/freq_of_adj/strided_slice:output:0@sequential_2/dense_features/freq_of_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:é
/sequential_2/dense_features/freq_of_adj/ReshapeReshape;sequential_2/dense_features/freq_of_adj/ExpandDims:output:0>sequential_2/dense_features/freq_of_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
6sequential_2/dense_features/freq_of_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ŕ
2sequential_2/dense_features/freq_of_adv/ExpandDims
ExpandDimsfreq_of_adv?sequential_2/dense_features/freq_of_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
-sequential_2/dense_features/freq_of_adv/ShapeShape;sequential_2/dense_features/freq_of_adv/ExpandDims:output:0*
T0*
_output_shapes
:
;sequential_2/dense_features/freq_of_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=sequential_2/dense_features/freq_of_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=sequential_2/dense_features/freq_of_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/dense_features/freq_of_adv/strided_sliceStridedSlice6sequential_2/dense_features/freq_of_adv/Shape:output:0Dsequential_2/dense_features/freq_of_adv/strided_slice/stack:output:0Fsequential_2/dense_features/freq_of_adv/strided_slice/stack_1:output:0Fsequential_2/dense_features/freq_of_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
7sequential_2/dense_features/freq_of_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :í
5sequential_2/dense_features/freq_of_adv/Reshape/shapePack>sequential_2/dense_features/freq_of_adv/strided_slice:output:0@sequential_2/dense_features/freq_of_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:é
/sequential_2/dense_features/freq_of_adv/ReshapeReshape;sequential_2/dense_features/freq_of_adv/ExpandDims:output:0>sequential_2/dense_features/freq_of_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
?sequential_2/dense_features/freq_of_distinct_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ű
;sequential_2/dense_features/freq_of_distinct_adj/ExpandDims
ExpandDimsfreq_of_distinct_adjHsequential_2/dense_features/freq_of_distinct_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
6sequential_2/dense_features/freq_of_distinct_adj/ShapeShapeDsequential_2/dense_features/freq_of_distinct_adj/ExpandDims:output:0*
T0*
_output_shapes
:
Dsequential_2/dense_features/freq_of_distinct_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Fsequential_2/dense_features/freq_of_distinct_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Fsequential_2/dense_features/freq_of_distinct_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ć
>sequential_2/dense_features/freq_of_distinct_adj/strided_sliceStridedSlice?sequential_2/dense_features/freq_of_distinct_adj/Shape:output:0Msequential_2/dense_features/freq_of_distinct_adj/strided_slice/stack:output:0Osequential_2/dense_features/freq_of_distinct_adj/strided_slice/stack_1:output:0Osequential_2/dense_features/freq_of_distinct_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
@sequential_2/dense_features/freq_of_distinct_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
>sequential_2/dense_features/freq_of_distinct_adj/Reshape/shapePackGsequential_2/dense_features/freq_of_distinct_adj/strided_slice:output:0Isequential_2/dense_features/freq_of_distinct_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
8sequential_2/dense_features/freq_of_distinct_adj/ReshapeReshapeDsequential_2/dense_features/freq_of_distinct_adj/ExpandDims:output:0Gsequential_2/dense_features/freq_of_distinct_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
?sequential_2/dense_features/freq_of_distinct_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ű
;sequential_2/dense_features/freq_of_distinct_adv/ExpandDims
ExpandDimsfreq_of_distinct_advHsequential_2/dense_features/freq_of_distinct_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
6sequential_2/dense_features/freq_of_distinct_adv/ShapeShapeDsequential_2/dense_features/freq_of_distinct_adv/ExpandDims:output:0*
T0*
_output_shapes
:
Dsequential_2/dense_features/freq_of_distinct_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Fsequential_2/dense_features/freq_of_distinct_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Fsequential_2/dense_features/freq_of_distinct_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ć
>sequential_2/dense_features/freq_of_distinct_adv/strided_sliceStridedSlice?sequential_2/dense_features/freq_of_distinct_adv/Shape:output:0Msequential_2/dense_features/freq_of_distinct_adv/strided_slice/stack:output:0Osequential_2/dense_features/freq_of_distinct_adv/strided_slice/stack_1:output:0Osequential_2/dense_features/freq_of_distinct_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
@sequential_2/dense_features/freq_of_distinct_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
>sequential_2/dense_features/freq_of_distinct_adv/Reshape/shapePackGsequential_2/dense_features/freq_of_distinct_adv/strided_slice:output:0Isequential_2/dense_features/freq_of_distinct_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
8sequential_2/dense_features/freq_of_distinct_adv/ReshapeReshapeDsequential_2/dense_features/freq_of_distinct_adv/ExpandDims:output:0Gsequential_2/dense_features/freq_of_distinct_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
7sequential_2/dense_features/freq_of_noun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ă
3sequential_2/dense_features/freq_of_noun/ExpandDims
ExpandDimsfreq_of_noun@sequential_2/dense_features/freq_of_noun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.sequential_2/dense_features/freq_of_noun/ShapeShape<sequential_2/dense_features/freq_of_noun/ExpandDims:output:0*
T0*
_output_shapes
:
<sequential_2/dense_features/freq_of_noun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>sequential_2/dense_features/freq_of_noun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>sequential_2/dense_features/freq_of_noun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6sequential_2/dense_features/freq_of_noun/strided_sliceStridedSlice7sequential_2/dense_features/freq_of_noun/Shape:output:0Esequential_2/dense_features/freq_of_noun/strided_slice/stack:output:0Gsequential_2/dense_features/freq_of_noun/strided_slice/stack_1:output:0Gsequential_2/dense_features/freq_of_noun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
8sequential_2/dense_features/freq_of_noun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :đ
6sequential_2/dense_features/freq_of_noun/Reshape/shapePack?sequential_2/dense_features/freq_of_noun/strided_slice:output:0Asequential_2/dense_features/freq_of_noun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ě
0sequential_2/dense_features/freq_of_noun/ReshapeReshape<sequential_2/dense_features/freq_of_noun/ExpandDims:output:0?sequential_2/dense_features/freq_of_noun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
:sequential_2/dense_features/freq_of_pronoun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ě
6sequential_2/dense_features/freq_of_pronoun/ExpandDims
ExpandDimsfreq_of_pronounCsequential_2/dense_features/freq_of_pronoun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
1sequential_2/dense_features/freq_of_pronoun/ShapeShape?sequential_2/dense_features/freq_of_pronoun/ExpandDims:output:0*
T0*
_output_shapes
:
?sequential_2/dense_features/freq_of_pronoun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Asequential_2/dense_features/freq_of_pronoun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Asequential_2/dense_features/freq_of_pronoun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
9sequential_2/dense_features/freq_of_pronoun/strided_sliceStridedSlice:sequential_2/dense_features/freq_of_pronoun/Shape:output:0Hsequential_2/dense_features/freq_of_pronoun/strided_slice/stack:output:0Jsequential_2/dense_features/freq_of_pronoun/strided_slice/stack_1:output:0Jsequential_2/dense_features/freq_of_pronoun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;sequential_2/dense_features/freq_of_pronoun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ů
9sequential_2/dense_features/freq_of_pronoun/Reshape/shapePackBsequential_2/dense_features/freq_of_pronoun/strided_slice:output:0Dsequential_2/dense_features/freq_of_pronoun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ő
3sequential_2/dense_features/freq_of_pronoun/ReshapeReshape?sequential_2/dense_features/freq_of_pronoun/ExpandDims:output:0Bsequential_2/dense_features/freq_of_pronoun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
=sequential_2/dense_features/freq_of_transition/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ő
9sequential_2/dense_features/freq_of_transition/ExpandDims
ExpandDimsfreq_of_transitionFsequential_2/dense_features/freq_of_transition/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
4sequential_2/dense_features/freq_of_transition/ShapeShapeBsequential_2/dense_features/freq_of_transition/ExpandDims:output:0*
T0*
_output_shapes
:
Bsequential_2/dense_features/freq_of_transition/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Dsequential_2/dense_features/freq_of_transition/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Dsequential_2/dense_features/freq_of_transition/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ź
<sequential_2/dense_features/freq_of_transition/strided_sliceStridedSlice=sequential_2/dense_features/freq_of_transition/Shape:output:0Ksequential_2/dense_features/freq_of_transition/strided_slice/stack:output:0Msequential_2/dense_features/freq_of_transition/strided_slice/stack_1:output:0Msequential_2/dense_features/freq_of_transition/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
>sequential_2/dense_features/freq_of_transition/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
<sequential_2/dense_features/freq_of_transition/Reshape/shapePackEsequential_2/dense_features/freq_of_transition/strided_slice:output:0Gsequential_2/dense_features/freq_of_transition/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ţ
6sequential_2/dense_features/freq_of_transition/ReshapeReshapeBsequential_2/dense_features/freq_of_transition/ExpandDims:output:0Esequential_2/dense_features/freq_of_transition/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
7sequential_2/dense_features/freq_of_verb/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ă
3sequential_2/dense_features/freq_of_verb/ExpandDims
ExpandDimsfreq_of_verb@sequential_2/dense_features/freq_of_verb/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.sequential_2/dense_features/freq_of_verb/ShapeShape<sequential_2/dense_features/freq_of_verb/ExpandDims:output:0*
T0*
_output_shapes
:
<sequential_2/dense_features/freq_of_verb/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>sequential_2/dense_features/freq_of_verb/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>sequential_2/dense_features/freq_of_verb/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6sequential_2/dense_features/freq_of_verb/strided_sliceStridedSlice7sequential_2/dense_features/freq_of_verb/Shape:output:0Esequential_2/dense_features/freq_of_verb/strided_slice/stack:output:0Gsequential_2/dense_features/freq_of_verb/strided_slice/stack_1:output:0Gsequential_2/dense_features/freq_of_verb/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
8sequential_2/dense_features/freq_of_verb/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :đ
6sequential_2/dense_features/freq_of_verb/Reshape/shapePack?sequential_2/dense_features/freq_of_verb/strided_slice:output:0Asequential_2/dense_features/freq_of_verb/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ě
0sequential_2/dense_features/freq_of_verb/ReshapeReshape<sequential_2/dense_features/freq_of_verb/ExpandDims:output:0?sequential_2/dense_features/freq_of_verb/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
>sequential_2/dense_features/freq_of_wrong_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ř
:sequential_2/dense_features/freq_of_wrong_words/ExpandDims
ExpandDimsfreq_of_wrong_wordsGsequential_2/dense_features/freq_of_wrong_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
5sequential_2/dense_features/freq_of_wrong_words/ShapeShapeCsequential_2/dense_features/freq_of_wrong_words/ExpandDims:output:0*
T0*
_output_shapes
:
Csequential_2/dense_features/freq_of_wrong_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Esequential_2/dense_features/freq_of_wrong_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Esequential_2/dense_features/freq_of_wrong_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Á
=sequential_2/dense_features/freq_of_wrong_words/strided_sliceStridedSlice>sequential_2/dense_features/freq_of_wrong_words/Shape:output:0Lsequential_2/dense_features/freq_of_wrong_words/strided_slice/stack:output:0Nsequential_2/dense_features/freq_of_wrong_words/strided_slice/stack_1:output:0Nsequential_2/dense_features/freq_of_wrong_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?sequential_2/dense_features/freq_of_wrong_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
=sequential_2/dense_features/freq_of_wrong_words/Reshape/shapePackFsequential_2/dense_features/freq_of_wrong_words/strided_slice:output:0Hsequential_2/dense_features/freq_of_wrong_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
7sequential_2/dense_features/freq_of_wrong_words/ReshapeReshapeCsequential_2/dense_features/freq_of_wrong_words/ExpandDims:output:0Fsequential_2/dense_features/freq_of_wrong_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
?sequential_2/dense_features/lexrank_avg_min_diff/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ű
;sequential_2/dense_features/lexrank_avg_min_diff/ExpandDims
ExpandDimslexrank_avg_min_diffHsequential_2/dense_features/lexrank_avg_min_diff/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
6sequential_2/dense_features/lexrank_avg_min_diff/ShapeShapeDsequential_2/dense_features/lexrank_avg_min_diff/ExpandDims:output:0*
T0*
_output_shapes
:
Dsequential_2/dense_features/lexrank_avg_min_diff/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Fsequential_2/dense_features/lexrank_avg_min_diff/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Fsequential_2/dense_features/lexrank_avg_min_diff/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ć
>sequential_2/dense_features/lexrank_avg_min_diff/strided_sliceStridedSlice?sequential_2/dense_features/lexrank_avg_min_diff/Shape:output:0Msequential_2/dense_features/lexrank_avg_min_diff/strided_slice/stack:output:0Osequential_2/dense_features/lexrank_avg_min_diff/strided_slice/stack_1:output:0Osequential_2/dense_features/lexrank_avg_min_diff/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
@sequential_2/dense_features/lexrank_avg_min_diff/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
>sequential_2/dense_features/lexrank_avg_min_diff/Reshape/shapePackGsequential_2/dense_features/lexrank_avg_min_diff/strided_slice:output:0Isequential_2/dense_features/lexrank_avg_min_diff/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
8sequential_2/dense_features/lexrank_avg_min_diff/ReshapeReshapeDsequential_2/dense_features/lexrank_avg_min_diff/ExpandDims:output:0Gsequential_2/dense_features/lexrank_avg_min_diff/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
@sequential_2/dense_features/lexrank_interquartile/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ţ
<sequential_2/dense_features/lexrank_interquartile/ExpandDims
ExpandDimslexrank_interquartileIsequential_2/dense_features/lexrank_interquartile/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
7sequential_2/dense_features/lexrank_interquartile/ShapeShapeEsequential_2/dense_features/lexrank_interquartile/ExpandDims:output:0*
T0*
_output_shapes
:
Esequential_2/dense_features/lexrank_interquartile/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Gsequential_2/dense_features/lexrank_interquartile/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Gsequential_2/dense_features/lexrank_interquartile/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ë
?sequential_2/dense_features/lexrank_interquartile/strided_sliceStridedSlice@sequential_2/dense_features/lexrank_interquartile/Shape:output:0Nsequential_2/dense_features/lexrank_interquartile/strided_slice/stack:output:0Psequential_2/dense_features/lexrank_interquartile/strided_slice/stack_1:output:0Psequential_2/dense_features/lexrank_interquartile/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Asequential_2/dense_features/lexrank_interquartile/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?sequential_2/dense_features/lexrank_interquartile/Reshape/shapePackHsequential_2/dense_features/lexrank_interquartile/strided_slice:output:0Jsequential_2/dense_features/lexrank_interquartile/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
9sequential_2/dense_features/lexrank_interquartile/ReshapeReshapeEsequential_2/dense_features/lexrank_interquartile/ExpandDims:output:0Hsequential_2/dense_features/lexrank_interquartile/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
9sequential_2/dense_features/mcalpine_eflaw/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙É
5sequential_2/dense_features/mcalpine_eflaw/ExpandDims
ExpandDimsmcalpine_eflawBsequential_2/dense_features/mcalpine_eflaw/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
0sequential_2/dense_features/mcalpine_eflaw/ShapeShape>sequential_2/dense_features/mcalpine_eflaw/ExpandDims:output:0*
T0*
_output_shapes
:
>sequential_2/dense_features/mcalpine_eflaw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
@sequential_2/dense_features/mcalpine_eflaw/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
@sequential_2/dense_features/mcalpine_eflaw/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¨
8sequential_2/dense_features/mcalpine_eflaw/strided_sliceStridedSlice9sequential_2/dense_features/mcalpine_eflaw/Shape:output:0Gsequential_2/dense_features/mcalpine_eflaw/strided_slice/stack:output:0Isequential_2/dense_features/mcalpine_eflaw/strided_slice/stack_1:output:0Isequential_2/dense_features/mcalpine_eflaw/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:sequential_2/dense_features/mcalpine_eflaw/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ö
8sequential_2/dense_features/mcalpine_eflaw/Reshape/shapePackAsequential_2/dense_features/mcalpine_eflaw/strided_slice:output:0Csequential_2/dense_features/mcalpine_eflaw/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ň
2sequential_2/dense_features/mcalpine_eflaw/ReshapeReshape>sequential_2/dense_features/mcalpine_eflaw/ExpandDims:output:0Asequential_2/dense_features/mcalpine_eflaw/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
6sequential_2/dense_features/noun_to_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ŕ
2sequential_2/dense_features/noun_to_adj/ExpandDims
ExpandDimsnoun_to_adj?sequential_2/dense_features/noun_to_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
-sequential_2/dense_features/noun_to_adj/ShapeShape;sequential_2/dense_features/noun_to_adj/ExpandDims:output:0*
T0*
_output_shapes
:
;sequential_2/dense_features/noun_to_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=sequential_2/dense_features/noun_to_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=sequential_2/dense_features/noun_to_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/dense_features/noun_to_adj/strided_sliceStridedSlice6sequential_2/dense_features/noun_to_adj/Shape:output:0Dsequential_2/dense_features/noun_to_adj/strided_slice/stack:output:0Fsequential_2/dense_features/noun_to_adj/strided_slice/stack_1:output:0Fsequential_2/dense_features/noun_to_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
7sequential_2/dense_features/noun_to_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :í
5sequential_2/dense_features/noun_to_adj/Reshape/shapePack>sequential_2/dense_features/noun_to_adj/strided_slice:output:0@sequential_2/dense_features/noun_to_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:é
/sequential_2/dense_features/noun_to_adj/ReshapeReshape;sequential_2/dense_features/noun_to_adj/ExpandDims:output:0>sequential_2/dense_features/noun_to_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
@sequential_2/dense_features/num_of_grammar_errors/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ţ
<sequential_2/dense_features/num_of_grammar_errors/ExpandDims
ExpandDimsnum_of_grammar_errorsIsequential_2/dense_features/num_of_grammar_errors/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ć
6sequential_2/dense_features/num_of_grammar_errors/CastCastEsequential_2/dense_features/num_of_grammar_errors/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ą
7sequential_2/dense_features/num_of_grammar_errors/ShapeShape:sequential_2/dense_features/num_of_grammar_errors/Cast:y:0*
T0*
_output_shapes
:
Esequential_2/dense_features/num_of_grammar_errors/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Gsequential_2/dense_features/num_of_grammar_errors/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Gsequential_2/dense_features/num_of_grammar_errors/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ë
?sequential_2/dense_features/num_of_grammar_errors/strided_sliceStridedSlice@sequential_2/dense_features/num_of_grammar_errors/Shape:output:0Nsequential_2/dense_features/num_of_grammar_errors/strided_slice/stack:output:0Psequential_2/dense_features/num_of_grammar_errors/strided_slice/stack_1:output:0Psequential_2/dense_features/num_of_grammar_errors/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Asequential_2/dense_features/num_of_grammar_errors/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
?sequential_2/dense_features/num_of_grammar_errors/Reshape/shapePackHsequential_2/dense_features/num_of_grammar_errors/strided_slice:output:0Jsequential_2/dense_features/num_of_grammar_errors/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ü
9sequential_2/dense_features/num_of_grammar_errors/ReshapeReshape:sequential_2/dense_features/num_of_grammar_errors/Cast:y:0Hsequential_2/dense_features/num_of_grammar_errors/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
=sequential_2/dense_features/num_of_short_forms/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ő
9sequential_2/dense_features/num_of_short_forms/ExpandDims
ExpandDimsnum_of_short_formsFsequential_2/dense_features/num_of_short_forms/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ
3sequential_2/dense_features/num_of_short_forms/CastCastBsequential_2/dense_features/num_of_short_forms/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
4sequential_2/dense_features/num_of_short_forms/ShapeShape7sequential_2/dense_features/num_of_short_forms/Cast:y:0*
T0*
_output_shapes
:
Bsequential_2/dense_features/num_of_short_forms/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Dsequential_2/dense_features/num_of_short_forms/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Dsequential_2/dense_features/num_of_short_forms/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ź
<sequential_2/dense_features/num_of_short_forms/strided_sliceStridedSlice=sequential_2/dense_features/num_of_short_forms/Shape:output:0Ksequential_2/dense_features/num_of_short_forms/strided_slice/stack:output:0Msequential_2/dense_features/num_of_short_forms/strided_slice/stack_1:output:0Msequential_2/dense_features/num_of_short_forms/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
>sequential_2/dense_features/num_of_short_forms/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
<sequential_2/dense_features/num_of_short_forms/Reshape/shapePackEsequential_2/dense_features/num_of_short_forms/strided_slice:output:0Gsequential_2/dense_features/num_of_short_forms/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ó
6sequential_2/dense_features/num_of_short_forms/ReshapeReshape7sequential_2/dense_features/num_of_short_forms/Cast:y:0Esequential_2/dense_features/num_of_short_forms/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
?sequential_2/dense_features/number_of_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ű
;sequential_2/dense_features/number_of_diff_words/ExpandDims
ExpandDimsnumber_of_diff_wordsHsequential_2/dense_features/number_of_diff_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ä
5sequential_2/dense_features/number_of_diff_words/CastCastDsequential_2/dense_features/number_of_diff_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
6sequential_2/dense_features/number_of_diff_words/ShapeShape9sequential_2/dense_features/number_of_diff_words/Cast:y:0*
T0*
_output_shapes
:
Dsequential_2/dense_features/number_of_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Fsequential_2/dense_features/number_of_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Fsequential_2/dense_features/number_of_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ć
>sequential_2/dense_features/number_of_diff_words/strided_sliceStridedSlice?sequential_2/dense_features/number_of_diff_words/Shape:output:0Msequential_2/dense_features/number_of_diff_words/strided_slice/stack:output:0Osequential_2/dense_features/number_of_diff_words/strided_slice/stack_1:output:0Osequential_2/dense_features/number_of_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
@sequential_2/dense_features/number_of_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
>sequential_2/dense_features/number_of_diff_words/Reshape/shapePackGsequential_2/dense_features/number_of_diff_words/strided_slice:output:0Isequential_2/dense_features/number_of_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ů
8sequential_2/dense_features/number_of_diff_words/ReshapeReshape9sequential_2/dense_features/number_of_diff_words/Cast:y:0Gsequential_2/dense_features/number_of_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
:sequential_2/dense_features/number_of_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ě
6sequential_2/dense_features/number_of_words/ExpandDims
ExpandDimsnumber_of_wordsCsequential_2/dense_features/number_of_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ş
0sequential_2/dense_features/number_of_words/CastCast?sequential_2/dense_features/number_of_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
1sequential_2/dense_features/number_of_words/ShapeShape4sequential_2/dense_features/number_of_words/Cast:y:0*
T0*
_output_shapes
:
?sequential_2/dense_features/number_of_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Asequential_2/dense_features/number_of_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Asequential_2/dense_features/number_of_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
9sequential_2/dense_features/number_of_words/strided_sliceStridedSlice:sequential_2/dense_features/number_of_words/Shape:output:0Hsequential_2/dense_features/number_of_words/strided_slice/stack:output:0Jsequential_2/dense_features/number_of_words/strided_slice/stack_1:output:0Jsequential_2/dense_features/number_of_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;sequential_2/dense_features/number_of_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ů
9sequential_2/dense_features/number_of_words/Reshape/shapePackBsequential_2/dense_features/number_of_words/strided_slice:output:0Dsequential_2/dense_features/number_of_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ę
3sequential_2/dense_features/number_of_words/ReshapeReshape4sequential_2/dense_features/number_of_words/Cast:y:0Bsequential_2/dense_features/number_of_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
;sequential_2/dense_features/phrase_diversity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ď
7sequential_2/dense_features/phrase_diversity/ExpandDims
ExpandDimsphrase_diversityDsequential_2/dense_features/phrase_diversity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙˘
2sequential_2/dense_features/phrase_diversity/ShapeShape@sequential_2/dense_features/phrase_diversity/ExpandDims:output:0*
T0*
_output_shapes
:
@sequential_2/dense_features/phrase_diversity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Bsequential_2/dense_features/phrase_diversity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Bsequential_2/dense_features/phrase_diversity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:˛
:sequential_2/dense_features/phrase_diversity/strided_sliceStridedSlice;sequential_2/dense_features/phrase_diversity/Shape:output:0Isequential_2/dense_features/phrase_diversity/strided_slice/stack:output:0Ksequential_2/dense_features/phrase_diversity/strided_slice/stack_1:output:0Ksequential_2/dense_features/phrase_diversity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask~
<sequential_2/dense_features/phrase_diversity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ü
:sequential_2/dense_features/phrase_diversity/Reshape/shapePackCsequential_2/dense_features/phrase_diversity/strided_slice:output:0Esequential_2/dense_features/phrase_diversity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ř
4sequential_2/dense_features/phrase_diversity/ReshapeReshape@sequential_2/dense_features/phrase_diversity/ExpandDims:output:0Csequential_2/dense_features/phrase_diversity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
7sequential_2/dense_features/punctuations/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ă
3sequential_2/dense_features/punctuations/ExpandDims
ExpandDimspunctuations@sequential_2/dense_features/punctuations/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
.sequential_2/dense_features/punctuations/ShapeShape<sequential_2/dense_features/punctuations/ExpandDims:output:0*
T0*
_output_shapes
:
<sequential_2/dense_features/punctuations/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>sequential_2/dense_features/punctuations/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>sequential_2/dense_features/punctuations/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6sequential_2/dense_features/punctuations/strided_sliceStridedSlice7sequential_2/dense_features/punctuations/Shape:output:0Esequential_2/dense_features/punctuations/strided_slice/stack:output:0Gsequential_2/dense_features/punctuations/strided_slice/stack_1:output:0Gsequential_2/dense_features/punctuations/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
8sequential_2/dense_features/punctuations/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :đ
6sequential_2/dense_features/punctuations/Reshape/shapePack?sequential_2/dense_features/punctuations/strided_slice:output:0Asequential_2/dense_features/punctuations/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ě
0sequential_2/dense_features/punctuations/ReshapeReshape<sequential_2/dense_features/punctuations/ExpandDims:output:0?sequential_2/dense_features/punctuations/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
>sequential_2/dense_features/sentence_complexity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ř
:sequential_2/dense_features/sentence_complexity/ExpandDims
ExpandDimssentence_complexityGsequential_2/dense_features/sentence_complexity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
5sequential_2/dense_features/sentence_complexity/ShapeShapeCsequential_2/dense_features/sentence_complexity/ExpandDims:output:0*
T0*
_output_shapes
:
Csequential_2/dense_features/sentence_complexity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Esequential_2/dense_features/sentence_complexity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Esequential_2/dense_features/sentence_complexity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Á
=sequential_2/dense_features/sentence_complexity/strided_sliceStridedSlice>sequential_2/dense_features/sentence_complexity/Shape:output:0Lsequential_2/dense_features/sentence_complexity/strided_slice/stack:output:0Nsequential_2/dense_features/sentence_complexity/strided_slice/stack_1:output:0Nsequential_2/dense_features/sentence_complexity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?sequential_2/dense_features/sentence_complexity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
=sequential_2/dense_features/sentence_complexity/Reshape/shapePackFsequential_2/dense_features/sentence_complexity/strided_slice:output:0Hsequential_2/dense_features/sentence_complexity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
7sequential_2/dense_features/sentence_complexity/ReshapeReshapeCsequential_2/dense_features/sentence_complexity/ExpandDims:output:0Fsequential_2/dense_features/sentence_complexity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
=sequential_2/dense_features/sentiment_compound/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ő
9sequential_2/dense_features/sentiment_compound/ExpandDims
ExpandDimssentiment_compoundFsequential_2/dense_features/sentiment_compound/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
4sequential_2/dense_features/sentiment_compound/ShapeShapeBsequential_2/dense_features/sentiment_compound/ExpandDims:output:0*
T0*
_output_shapes
:
Bsequential_2/dense_features/sentiment_compound/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Dsequential_2/dense_features/sentiment_compound/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Dsequential_2/dense_features/sentiment_compound/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ź
<sequential_2/dense_features/sentiment_compound/strided_sliceStridedSlice=sequential_2/dense_features/sentiment_compound/Shape:output:0Ksequential_2/dense_features/sentiment_compound/strided_slice/stack:output:0Msequential_2/dense_features/sentiment_compound/strided_slice/stack_1:output:0Msequential_2/dense_features/sentiment_compound/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
>sequential_2/dense_features/sentiment_compound/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
<sequential_2/dense_features/sentiment_compound/Reshape/shapePackEsequential_2/dense_features/sentiment_compound/strided_slice:output:0Gsequential_2/dense_features/sentiment_compound/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ţ
6sequential_2/dense_features/sentiment_compound/ReshapeReshapeBsequential_2/dense_features/sentiment_compound/ExpandDims:output:0Esequential_2/dense_features/sentiment_compound/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
=sequential_2/dense_features/sentiment_negative/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ő
9sequential_2/dense_features/sentiment_negative/ExpandDims
ExpandDimssentiment_negativeFsequential_2/dense_features/sentiment_negative/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
4sequential_2/dense_features/sentiment_negative/ShapeShapeBsequential_2/dense_features/sentiment_negative/ExpandDims:output:0*
T0*
_output_shapes
:
Bsequential_2/dense_features/sentiment_negative/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Dsequential_2/dense_features/sentiment_negative/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Dsequential_2/dense_features/sentiment_negative/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ź
<sequential_2/dense_features/sentiment_negative/strided_sliceStridedSlice=sequential_2/dense_features/sentiment_negative/Shape:output:0Ksequential_2/dense_features/sentiment_negative/strided_slice/stack:output:0Msequential_2/dense_features/sentiment_negative/strided_slice/stack_1:output:0Msequential_2/dense_features/sentiment_negative/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
>sequential_2/dense_features/sentiment_negative/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
<sequential_2/dense_features/sentiment_negative/Reshape/shapePackEsequential_2/dense_features/sentiment_negative/strided_slice:output:0Gsequential_2/dense_features/sentiment_negative/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ţ
6sequential_2/dense_features/sentiment_negative/ReshapeReshapeBsequential_2/dense_features/sentiment_negative/ExpandDims:output:0Esequential_2/dense_features/sentiment_negative/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
=sequential_2/dense_features/sentiment_positive/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ő
9sequential_2/dense_features/sentiment_positive/ExpandDims
ExpandDimssentiment_positiveFsequential_2/dense_features/sentiment_positive/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
4sequential_2/dense_features/sentiment_positive/ShapeShapeBsequential_2/dense_features/sentiment_positive/ExpandDims:output:0*
T0*
_output_shapes
:
Bsequential_2/dense_features/sentiment_positive/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Dsequential_2/dense_features/sentiment_positive/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Dsequential_2/dense_features/sentiment_positive/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ź
<sequential_2/dense_features/sentiment_positive/strided_sliceStridedSlice=sequential_2/dense_features/sentiment_positive/Shape:output:0Ksequential_2/dense_features/sentiment_positive/strided_slice/stack:output:0Msequential_2/dense_features/sentiment_positive/strided_slice/stack_1:output:0Msequential_2/dense_features/sentiment_positive/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
>sequential_2/dense_features/sentiment_positive/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
<sequential_2/dense_features/sentiment_positive/Reshape/shapePackEsequential_2/dense_features/sentiment_positive/strided_slice:output:0Gsequential_2/dense_features/sentiment_positive/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ţ
6sequential_2/dense_features/sentiment_positive/ReshapeReshapeBsequential_2/dense_features/sentiment_positive/ExpandDims:output:0Esequential_2/dense_features/sentiment_positive/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
>sequential_2/dense_features/stopwords_frequency/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ř
:sequential_2/dense_features/stopwords_frequency/ExpandDims
ExpandDimsstopwords_frequencyGsequential_2/dense_features/stopwords_frequency/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙¨
5sequential_2/dense_features/stopwords_frequency/ShapeShapeCsequential_2/dense_features/stopwords_frequency/ExpandDims:output:0*
T0*
_output_shapes
:
Csequential_2/dense_features/stopwords_frequency/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Esequential_2/dense_features/stopwords_frequency/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Esequential_2/dense_features/stopwords_frequency/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Á
=sequential_2/dense_features/stopwords_frequency/strided_sliceStridedSlice>sequential_2/dense_features/stopwords_frequency/Shape:output:0Lsequential_2/dense_features/stopwords_frequency/strided_slice/stack:output:0Nsequential_2/dense_features/stopwords_frequency/strided_slice/stack_1:output:0Nsequential_2/dense_features/stopwords_frequency/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?sequential_2/dense_features/stopwords_frequency/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
=sequential_2/dense_features/stopwords_frequency/Reshape/shapePackFsequential_2/dense_features/stopwords_frequency/strided_slice:output:0Hsequential_2/dense_features/stopwords_frequency/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
7sequential_2/dense_features/stopwords_frequency/ReshapeReshapeCsequential_2/dense_features/stopwords_frequency/ExpandDims:output:0Fsequential_2/dense_features/stopwords_frequency/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
8sequential_2/dense_features/text_standard/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ć
4sequential_2/dense_features/text_standard/ExpandDims
ExpandDimstext_standardAsequential_2/dense_features/text_standard/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
/sequential_2/dense_features/text_standard/ShapeShape=sequential_2/dense_features/text_standard/ExpandDims:output:0*
T0*
_output_shapes
:
=sequential_2/dense_features/text_standard/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?sequential_2/dense_features/text_standard/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?sequential_2/dense_features/text_standard/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ł
7sequential_2/dense_features/text_standard/strided_sliceStridedSlice8sequential_2/dense_features/text_standard/Shape:output:0Fsequential_2/dense_features/text_standard/strided_slice/stack:output:0Hsequential_2/dense_features/text_standard/strided_slice/stack_1:output:0Hsequential_2/dense_features/text_standard/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask{
9sequential_2/dense_features/text_standard/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ó
7sequential_2/dense_features/text_standard/Reshape/shapePack@sequential_2/dense_features/text_standard/strided_slice:output:0Bsequential_2/dense_features/text_standard/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ď
1sequential_2/dense_features/text_standard/ReshapeReshape=sequential_2/dense_features/text_standard/ExpandDims:output:0@sequential_2/dense_features/text_standard/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙y
.sequential_2/dense_features/ttr/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙¨
*sequential_2/dense_features/ttr/ExpandDims
ExpandDimsttr7sequential_2/dense_features/ttr/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
%sequential_2/dense_features/ttr/ShapeShape3sequential_2/dense_features/ttr/ExpandDims:output:0*
T0*
_output_shapes
:}
3sequential_2/dense_features/ttr/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential_2/dense_features/ttr/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/dense_features/ttr/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ń
-sequential_2/dense_features/ttr/strided_sliceStridedSlice.sequential_2/dense_features/ttr/Shape:output:0<sequential_2/dense_features/ttr/strided_slice/stack:output:0>sequential_2/dense_features/ttr/strided_slice/stack_1:output:0>sequential_2/dense_features/ttr/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/sequential_2/dense_features/ttr/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ő
-sequential_2/dense_features/ttr/Reshape/shapePack6sequential_2/dense_features/ttr/strided_slice:output:08sequential_2/dense_features/ttr/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ń
'sequential_2/dense_features/ttr/ReshapeReshape3sequential_2/dense_features/ttr/ExpandDims:output:06sequential_2/dense_features/ttr/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
6sequential_2/dense_features/verb_to_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ŕ
2sequential_2/dense_features/verb_to_adv/ExpandDims
ExpandDimsverb_to_adv?sequential_2/dense_features/verb_to_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
-sequential_2/dense_features/verb_to_adv/ShapeShape;sequential_2/dense_features/verb_to_adv/ExpandDims:output:0*
T0*
_output_shapes
:
;sequential_2/dense_features/verb_to_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=sequential_2/dense_features/verb_to_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=sequential_2/dense_features/verb_to_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5sequential_2/dense_features/verb_to_adv/strided_sliceStridedSlice6sequential_2/dense_features/verb_to_adv/Shape:output:0Dsequential_2/dense_features/verb_to_adv/strided_slice/stack:output:0Fsequential_2/dense_features/verb_to_adv/strided_slice/stack_1:output:0Fsequential_2/dense_features/verb_to_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
7sequential_2/dense_features/verb_to_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :í
5sequential_2/dense_features/verb_to_adv/Reshape/shapePack>sequential_2/dense_features/verb_to_adv/strided_slice:output:0@sequential_2/dense_features/verb_to_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:é
/sequential_2/dense_features/verb_to_adv/ReshapeReshape;sequential_2/dense_features/verb_to_adv/ExpandDims:output:0>sequential_2/dense_features/verb_to_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
'sequential_2/dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙¸
"sequential_2/dense_features/concatConcatV20sequential_2/dense_features/ARI/Reshape:output:0Asequential_2/dense_features/Incorrect_form_ratio/Reshape:output:0<sequential_2/dense_features/av_word_per_sen/Reshape:output:0<sequential_2/dense_features/coherence_score/Reshape:output:0Isequential_2/dense_features/dale_chall_readability_score/Reshape:output:0Asequential_2/dense_features/flesch_kincaid_grade/Reshape:output:0@sequential_2/dense_features/flesch_reading_ease/Reshape:output:0<sequential_2/dense_features/freq_diff_words/Reshape:output:08sequential_2/dense_features/freq_of_adj/Reshape:output:08sequential_2/dense_features/freq_of_adv/Reshape:output:0Asequential_2/dense_features/freq_of_distinct_adj/Reshape:output:0Asequential_2/dense_features/freq_of_distinct_adv/Reshape:output:09sequential_2/dense_features/freq_of_noun/Reshape:output:0<sequential_2/dense_features/freq_of_pronoun/Reshape:output:0?sequential_2/dense_features/freq_of_transition/Reshape:output:09sequential_2/dense_features/freq_of_verb/Reshape:output:0@sequential_2/dense_features/freq_of_wrong_words/Reshape:output:0Asequential_2/dense_features/lexrank_avg_min_diff/Reshape:output:0Bsequential_2/dense_features/lexrank_interquartile/Reshape:output:0;sequential_2/dense_features/mcalpine_eflaw/Reshape:output:08sequential_2/dense_features/noun_to_adj/Reshape:output:0Bsequential_2/dense_features/num_of_grammar_errors/Reshape:output:0?sequential_2/dense_features/num_of_short_forms/Reshape:output:0Asequential_2/dense_features/number_of_diff_words/Reshape:output:0<sequential_2/dense_features/number_of_words/Reshape:output:0=sequential_2/dense_features/phrase_diversity/Reshape:output:09sequential_2/dense_features/punctuations/Reshape:output:0@sequential_2/dense_features/sentence_complexity/Reshape:output:0?sequential_2/dense_features/sentiment_compound/Reshape:output:0?sequential_2/dense_features/sentiment_negative/Reshape:output:0?sequential_2/dense_features/sentiment_positive/Reshape:output:0@sequential_2/dense_features/stopwords_frequency/Reshape:output:0:sequential_2/dense_features/text_standard/Reshape:output:00sequential_2/dense_features/ttr/Reshape:output:08sequential_2/dense_features/verb_to_adv/Reshape:output:00sequential_2/dense_features/concat/axis:output:0*
N#*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙#
*sequential_2/Hidden0/MatMul/ReadVariableOpReadVariableOp3sequential_2_hidden0_matmul_readvariableop_resource*
_output_shapes

:#*
dtype0¸
sequential_2/Hidden0/MatMulMatMul+sequential_2/dense_features/concat:output:02sequential_2/Hidden0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+sequential_2/Hidden0/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_hidden0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ľ
sequential_2/Hidden0/BiasAddBiasAdd%sequential_2/Hidden0/MatMul:product:03sequential_2/Hidden0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙z
sequential_2/Hidden0/ReluRelu%sequential_2/Hidden0/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*sequential_2/Hidden1/MatMul/ReadVariableOpReadVariableOp3sequential_2_hidden1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0´
sequential_2/Hidden1/MatMulMatMul'sequential_2/Hidden0/Relu:activations:02sequential_2/Hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
+sequential_2/Hidden1/BiasAdd/ReadVariableOpReadVariableOp4sequential_2_hidden1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ľ
sequential_2/Hidden1/BiasAddBiasAdd%sequential_2/Hidden1/MatMul:product:03sequential_2/Hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙z
sequential_2/Hidden1/ReluRelu%sequential_2/Hidden1/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)sequential_2/Output/MatMul/ReadVariableOpReadVariableOp2sequential_2_output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0˛
sequential_2/Output/MatMulMatMul'sequential_2/Hidden1/Relu:activations:01sequential_2/Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*sequential_2/Output/BiasAdd/ReadVariableOpReadVariableOp3sequential_2_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0˛
sequential_2/Output/BiasAddBiasAdd$sequential_2/Output/MatMul:product:02sequential_2/Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙s
IdentityIdentity$sequential_2/Output/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ő
NoOpNoOp,^sequential_2/Hidden0/BiasAdd/ReadVariableOp+^sequential_2/Hidden0/MatMul/ReadVariableOp,^sequential_2/Hidden1/BiasAdd/ReadVariableOp+^sequential_2/Hidden1/MatMul/ReadVariableOp+^sequential_2/Output/BiasAdd/ReadVariableOp*^sequential_2/Output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ě
_input_shapesş
ˇ:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : 2Z
+sequential_2/Hidden0/BiasAdd/ReadVariableOp+sequential_2/Hidden0/BiasAdd/ReadVariableOp2X
*sequential_2/Hidden0/MatMul/ReadVariableOp*sequential_2/Hidden0/MatMul/ReadVariableOp2Z
+sequential_2/Hidden1/BiasAdd/ReadVariableOp+sequential_2/Hidden1/BiasAdd/ReadVariableOp2X
*sequential_2/Hidden1/MatMul/ReadVariableOp*sequential_2/Hidden1/MatMul/ReadVariableOp2X
*sequential_2/Output/BiasAdd/ReadVariableOp*sequential_2/Output/BiasAdd/ReadVariableOp2V
)sequential_2/Output/MatMul/ReadVariableOp)sequential_2/Output/MatMul/ReadVariableOp:H D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameARI:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameIncorrect_form_ratio:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameav_word_per_sen:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namecoherence_score:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
cohesion:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namecorrected_text:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namedale_chall_readability_score:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameflesch_kincaid_grade:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameflesch_reading_ease:T	P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefreq_diff_words:P
L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefreq_of_adj:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefreq_of_adv:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefreq_of_distinct_adj:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefreq_of_distinct_adv:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefreq_of_noun:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefreq_of_pronoun:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namefreq_of_transition:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefreq_of_verb:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namefreq_of_wrong_words:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namelexrank_avg_min_diff:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namelexrank_interquartile:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namemcalpine_eflaw:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namenoun_to_adj:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namenum_of_grammar_errors:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namenum_of_short_forms:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namenumber_of_diff_words:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namenumber_of_words:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namephrase_diversity:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepunctuations:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namesentence_complexity:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_compound:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_negative:W S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_positive:X!T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namestopwords_frequency:R"N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nametext_standard:H#D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namettr:P$L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameverb_to_adv
Ś5
Ż	
.__inference_dense_features_layer_call_fn_53433
features_ari	!
features_incorrect_form_ratio
features_av_word_per_sen
features_coherence_score
features_cohesion
features_corrected_text)
%features_dale_chall_readability_score!
features_flesch_kincaid_grade 
features_flesch_reading_ease
features_freq_diff_words
features_freq_of_adj
features_freq_of_adv!
features_freq_of_distinct_adj!
features_freq_of_distinct_adv
features_freq_of_noun
features_freq_of_pronoun
features_freq_of_transition
features_freq_of_verb 
features_freq_of_wrong_words!
features_lexrank_avg_min_diff"
features_lexrank_interquartile
features_mcalpine_eflaw
features_noun_to_adj"
features_num_of_grammar_errors	
features_num_of_short_forms	!
features_number_of_diff_words	
features_number_of_words	
features_phrase_diversity
features_punctuations 
features_sentence_complexity
features_sentiment_compound
features_sentiment_negative
features_sentiment_positive 
features_stopwords_frequency
features_text_standard
features_ttr
features_verb_to_adv
identity­

PartitionedCallPartitionedCallfeatures_arifeatures_incorrect_form_ratiofeatures_av_word_per_senfeatures_coherence_scorefeatures_cohesionfeatures_corrected_text%features_dale_chall_readability_scorefeatures_flesch_kincaid_gradefeatures_flesch_reading_easefeatures_freq_diff_wordsfeatures_freq_of_adjfeatures_freq_of_advfeatures_freq_of_distinct_adjfeatures_freq_of_distinct_advfeatures_freq_of_nounfeatures_freq_of_pronounfeatures_freq_of_transitionfeatures_freq_of_verbfeatures_freq_of_wrong_wordsfeatures_lexrank_avg_min_difffeatures_lexrank_interquartilefeatures_mcalpine_eflawfeatures_noun_to_adjfeatures_num_of_grammar_errorsfeatures_num_of_short_formsfeatures_number_of_diff_wordsfeatures_number_of_wordsfeatures_phrase_diversityfeatures_punctuationsfeatures_sentence_complexityfeatures_sentiment_compoundfeatures_sentiment_negativefeatures_sentiment_positivefeatures_stopwords_frequencyfeatures_text_standardfeatures_ttrfeatures_verb_to_adv*0
Tin)
'2%					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙#* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_51486`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙#"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ŕ
_input_shapesŽ
Ť:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:Q M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefeatures/ARI:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/Incorrect_form_ratio:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/av_word_per_sen:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/coherence_score:VR
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_namefeatures/cohesion:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_namefeatures/corrected_text:jf
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
?
_user_specified_name'%features/dale_chall_readability_score:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/flesch_kincaid_grade:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/flesch_reading_ease:]	Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/freq_diff_words:Y
U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/freq_of_adj:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/freq_of_adv:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/freq_of_distinct_adj:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/freq_of_distinct_adv:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namefeatures/freq_of_noun:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/freq_of_pronoun:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/freq_of_transition:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namefeatures/freq_of_verb:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/freq_of_wrong_words:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/lexrank_avg_min_diff:c_
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name features/lexrank_interquartile:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_namefeatures/mcalpine_eflaw:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/noun_to_adj:c_
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name features/num_of_grammar_errors:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/num_of_short_forms:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/number_of_diff_words:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/number_of_words:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_namefeatures/phrase_diversity:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namefeatures/punctuations:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/sentence_complexity:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/sentiment_compound:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/sentiment_negative:` \
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/sentiment_positive:a!]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/stopwords_frequency:["W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_namefeatures/text_standard:Q#M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefeatures/ttr:Y$U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/verb_to_adv
Ô5

G__inference_sequential_2_layer_call_and_return_conditional_losses_52211

inputs	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
	inputs_20
	inputs_21
	inputs_22
	inputs_23	
	inputs_24	
	inputs_25	
	inputs_26	
	inputs_27
	inputs_28
	inputs_29
	inputs_30
	inputs_31
	inputs_32
	inputs_33
	inputs_34
	inputs_35
	inputs_36
hidden0_52195:#
hidden0_52197:
hidden1_52200:
hidden1_52202:
output_52205:
output_52207:
identity˘Hidden0/StatefulPartitionedCall˘Hidden1/StatefulPartitionedCall˘Output/StatefulPartitionedCallę
dense_features/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19	inputs_20	inputs_21	inputs_22	inputs_23	inputs_24	inputs_25	inputs_26	inputs_27	inputs_28	inputs_29	inputs_30	inputs_31	inputs_32	inputs_33	inputs_34	inputs_35	inputs_36*0
Tin)
'2%					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙#* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_52061
Hidden0/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0hidden0_52195hidden0_52197*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Hidden0_layer_call_and_return_conditional_losses_51499
Hidden1/StatefulPartitionedCallStatefulPartitionedCall(Hidden0/StatefulPartitionedCall:output:0hidden1_52200hidden1_52202*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Hidden1_layer_call_and_return_conditional_losses_51516
Output/StatefulPartitionedCallStatefulPartitionedCall(Hidden1/StatefulPartitionedCall:output:0output_52205output_52207*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_51532v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
NoOpNoOp ^Hidden0/StatefulPartitionedCall ^Hidden1/StatefulPartitionedCall^Output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ě
_input_shapesş
ˇ:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : 2B
Hidden0/StatefulPartitionedCallHidden0/StatefulPartitionedCall2B
Hidden1/StatefulPartitionedCallHidden1/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K	G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K
G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K!G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K"G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K#G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K$G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ôv
˝
!__inference__traced_restore_54570
file_prefix>
,assignvariableop_sequential_2_hidden0_kernel:#:
,assignvariableop_1_sequential_2_hidden0_bias:@
.assignvariableop_2_sequential_2_hidden1_kernel::
,assignvariableop_3_sequential_2_hidden1_bias:?
-assignvariableop_4_sequential_2_output_kernel:9
+assignvariableop_5_sequential_2_output_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: %
assignvariableop_11_total_2: %
assignvariableop_12_count_2: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: #
assignvariableop_15_total: #
assignvariableop_16_count: H
6assignvariableop_17_adam_sequential_2_hidden0_kernel_m:#B
4assignvariableop_18_adam_sequential_2_hidden0_bias_m:H
6assignvariableop_19_adam_sequential_2_hidden1_kernel_m:B
4assignvariableop_20_adam_sequential_2_hidden1_bias_m:G
5assignvariableop_21_adam_sequential_2_output_kernel_m:A
3assignvariableop_22_adam_sequential_2_output_bias_m:H
6assignvariableop_23_adam_sequential_2_hidden0_kernel_v:#B
4assignvariableop_24_adam_sequential_2_hidden0_bias_v:H
6assignvariableop_25_adam_sequential_2_hidden1_kernel_v:B
4assignvariableop_26_adam_sequential_2_hidden1_bias_v:G
5assignvariableop_27_adam_sequential_2_output_kernel_v:A
3assignvariableop_28_adam_sequential_2_output_bias_v:
identity_30˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_10˘AssignVariableOp_11˘AssignVariableOp_12˘AssignVariableOp_13˘AssignVariableOp_14˘AssignVariableOp_15˘AssignVariableOp_16˘AssignVariableOp_17˘AssignVariableOp_18˘AssignVariableOp_19˘AssignVariableOp_2˘AssignVariableOp_20˘AssignVariableOp_21˘AssignVariableOp_22˘AssignVariableOp_23˘AssignVariableOp_24˘AssignVariableOp_25˘AssignVariableOp_26˘AssignVariableOp_27˘AssignVariableOp_28˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9ä
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBýB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHŹ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ľ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp,assignvariableop_sequential_2_hidden0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp,assignvariableop_1_sequential_2_hidden0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp.assignvariableop_2_sequential_2_hidden1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp,assignvariableop_3_sequential_2_hidden1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp-assignvariableop_4_sequential_2_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp+assignvariableop_5_sequential_2_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_sequential_2_hidden0_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_sequential_2_hidden0_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_sequential_2_hidden1_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_sequential_2_hidden1_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ś
AssignVariableOp_21AssignVariableOp5assignvariableop_21_adam_sequential_2_output_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_22AssignVariableOp3assignvariableop_22_adam_sequential_2_output_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adam_sequential_2_hidden0_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_sequential_2_hidden0_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_sequential_2_hidden1_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ľ
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_sequential_2_hidden1_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ś
AssignVariableOp_27AssignVariableOp5assignvariableop_27_adam_sequential_2_output_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_28AssignVariableOp3assignvariableop_28_adam_sequential_2_output_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Í
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_30IdentityIdentity_29:output:0^NoOp_1*
T0*
_output_shapes
: ş
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_30Identity_30:output:0*O
_input_shapes>
<: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
1

,__inference_sequential_2_layer_call_fn_51554
ari	
incorrect_form_ratio
av_word_per_sen
coherence_score
cohesion
corrected_text 
dale_chall_readability_score
flesch_kincaid_grade
flesch_reading_ease
freq_diff_words
freq_of_adj
freq_of_adv
freq_of_distinct_adj
freq_of_distinct_adv
freq_of_noun
freq_of_pronoun
freq_of_transition
freq_of_verb
freq_of_wrong_words
lexrank_avg_min_diff
lexrank_interquartile
mcalpine_eflaw
noun_to_adj
num_of_grammar_errors	
num_of_short_forms	
number_of_diff_words	
number_of_words	
phrase_diversity
punctuations
sentence_complexity
sentiment_compound
sentiment_negative
sentiment_positive
stopwords_frequency
text_standard
ttr
verb_to_adv
unknown:#
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity˘StatefulPartitionedCallź
StatefulPartitionedCallStatefulPartitionedCallariincorrect_form_ratioav_word_per_sencoherence_scorecohesioncorrected_textdale_chall_readability_scoreflesch_kincaid_gradeflesch_reading_easefreq_diff_wordsfreq_of_adjfreq_of_advfreq_of_distinct_adjfreq_of_distinct_advfreq_of_nounfreq_of_pronounfreq_of_transitionfreq_of_verbfreq_of_wrong_wordslexrank_avg_min_difflexrank_interquartilemcalpine_eflawnoun_to_adjnum_of_grammar_errorsnum_of_short_formsnumber_of_diff_wordsnumber_of_wordsphrase_diversitypunctuationssentence_complexitysentiment_compoundsentiment_negativesentiment_positivestopwords_frequencytext_standardttrverb_to_advunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*6
Tin/
-2+					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

%&'()**-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_51539o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ě
_input_shapesş
ˇ:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameARI:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameIncorrect_form_ratio:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameav_word_per_sen:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namecoherence_score:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
cohesion:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namecorrected_text:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namedale_chall_readability_score:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameflesch_kincaid_grade:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameflesch_reading_ease:T	P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefreq_diff_words:P
L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefreq_of_adj:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefreq_of_adv:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefreq_of_distinct_adj:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefreq_of_distinct_adv:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefreq_of_noun:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefreq_of_pronoun:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namefreq_of_transition:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefreq_of_verb:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namefreq_of_wrong_words:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namelexrank_avg_min_diff:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namelexrank_interquartile:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namemcalpine_eflaw:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namenoun_to_adj:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namenum_of_grammar_errors:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namenum_of_short_forms:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namenumber_of_diff_words:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namenumber_of_words:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namephrase_diversity:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepunctuations:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namesentence_complexity:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_compound:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_negative:W S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_positive:X!T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namestopwords_frequency:R"N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nametext_standard:H#D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namettr:P$L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameverb_to_adv
ť
Â
I__inference_dense_features_layer_call_and_return_conditional_losses_52061
features	

features_1

features_2

features_3

features_4

features_5

features_6

features_7

features_8

features_9
features_10
features_11
features_12
features_13
features_14
features_15
features_16
features_17
features_18
features_19
features_20
features_21
features_22
features_23	
features_24	
features_25	
features_26	
features_27
features_28
features_29
features_30
features_31
features_32
features_33
features_34
features_35
features_36
identity]
ARI/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙u
ARI/ExpandDims
ExpandDimsfeaturesARI/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ARI/CastCastARI/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙E
	ARI/ShapeShapeARI/Cast:y:0*
T0*
_output_shapes
:a
ARI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
ARI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
ARI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
ARI/strided_sliceStridedSliceARI/Shape:output:0 ARI/strided_slice/stack:output:0"ARI/strided_slice/stack_1:output:0"ARI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
ARI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ARI/Reshape/shapePackARI/strided_slice:output:0ARI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
ARI/ReshapeReshapeARI/Cast:y:0ARI/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#Incorrect_form_ratio/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
Incorrect_form_ratio/ExpandDims
ExpandDims
features_1,Incorrect_form_ratio/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
Incorrect_form_ratio/ShapeShape(Incorrect_form_ratio/ExpandDims:output:0*
T0*
_output_shapes
:r
(Incorrect_form_ratio/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Incorrect_form_ratio/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Incorrect_form_ratio/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"Incorrect_form_ratio/strided_sliceStridedSlice#Incorrect_form_ratio/Shape:output:01Incorrect_form_ratio/strided_slice/stack:output:03Incorrect_form_ratio/strided_slice/stack_1:output:03Incorrect_form_ratio/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Incorrect_form_ratio/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"Incorrect_form_ratio/Reshape/shapePack+Incorrect_form_ratio/strided_slice:output:0-Incorrect_form_ratio/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
Incorrect_form_ratio/ReshapeReshape(Incorrect_form_ratio/ExpandDims:output:0+Incorrect_form_ratio/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
av_word_per_sen/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
av_word_per_sen/ExpandDims
ExpandDims
features_2'av_word_per_sen/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
av_word_per_sen/ShapeShape#av_word_per_sen/ExpandDims:output:0*
T0*
_output_shapes
:m
#av_word_per_sen/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%av_word_per_sen/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%av_word_per_sen/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
av_word_per_sen/strided_sliceStridedSliceav_word_per_sen/Shape:output:0,av_word_per_sen/strided_slice/stack:output:0.av_word_per_sen/strided_slice/stack_1:output:0.av_word_per_sen/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
av_word_per_sen/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
av_word_per_sen/Reshape/shapePack&av_word_per_sen/strided_slice:output:0(av_word_per_sen/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
av_word_per_sen/ReshapeReshape#av_word_per_sen/ExpandDims:output:0&av_word_per_sen/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
coherence_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
coherence_score/ExpandDims
ExpandDims
features_3'coherence_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
coherence_score/ShapeShape#coherence_score/ExpandDims:output:0*
T0*
_output_shapes
:m
#coherence_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%coherence_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%coherence_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
coherence_score/strided_sliceStridedSlicecoherence_score/Shape:output:0,coherence_score/strided_slice/stack:output:0.coherence_score/strided_slice/stack_1:output:0.coherence_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
coherence_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
coherence_score/Reshape/shapePack&coherence_score/strided_slice:output:0(coherence_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
coherence_score/ReshapeReshape#coherence_score/ExpandDims:output:0&coherence_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙v
+dale_chall_readability_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Š
'dale_chall_readability_score/ExpandDims
ExpandDims
features_64dale_chall_readability_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"dale_chall_readability_score/ShapeShape0dale_chall_readability_score/ExpandDims:output:0*
T0*
_output_shapes
:z
0dale_chall_readability_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dale_chall_readability_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dale_chall_readability_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dale_chall_readability_score/strided_sliceStridedSlice+dale_chall_readability_score/Shape:output:09dale_chall_readability_score/strided_slice/stack:output:0;dale_chall_readability_score/strided_slice/stack_1:output:0;dale_chall_readability_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,dale_chall_readability_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ě
*dale_chall_readability_score/Reshape/shapePack3dale_chall_readability_score/strided_slice:output:05dale_chall_readability_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Č
$dale_chall_readability_score/ReshapeReshape0dale_chall_readability_score/ExpandDims:output:03dale_chall_readability_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#flesch_kincaid_grade/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
flesch_kincaid_grade/ExpandDims
ExpandDims
features_7,flesch_kincaid_grade/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
flesch_kincaid_grade/ShapeShape(flesch_kincaid_grade/ExpandDims:output:0*
T0*
_output_shapes
:r
(flesch_kincaid_grade/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*flesch_kincaid_grade/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*flesch_kincaid_grade/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"flesch_kincaid_grade/strided_sliceStridedSlice#flesch_kincaid_grade/Shape:output:01flesch_kincaid_grade/strided_slice/stack:output:03flesch_kincaid_grade/strided_slice/stack_1:output:03flesch_kincaid_grade/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$flesch_kincaid_grade/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"flesch_kincaid_grade/Reshape/shapePack+flesch_kincaid_grade/strided_slice:output:0-flesch_kincaid_grade/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
flesch_kincaid_grade/ReshapeReshape(flesch_kincaid_grade/ExpandDims:output:0+flesch_kincaid_grade/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"flesch_reading_ease/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
flesch_reading_ease/ExpandDims
ExpandDims
features_8+flesch_reading_ease/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
flesch_reading_ease/ShapeShape'flesch_reading_ease/ExpandDims:output:0*
T0*
_output_shapes
:q
'flesch_reading_ease/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)flesch_reading_ease/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)flesch_reading_ease/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!flesch_reading_ease/strided_sliceStridedSlice"flesch_reading_ease/Shape:output:00flesch_reading_ease/strided_slice/stack:output:02flesch_reading_ease/strided_slice/stack_1:output:02flesch_reading_ease/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#flesch_reading_ease/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!flesch_reading_ease/Reshape/shapePack*flesch_reading_ease/strided_slice:output:0,flesch_reading_ease/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
flesch_reading_ease/ReshapeReshape'flesch_reading_ease/ExpandDims:output:0*flesch_reading_ease/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
freq_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_diff_words/ExpandDims
ExpandDims
features_9'freq_diff_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
freq_diff_words/ShapeShape#freq_diff_words/ExpandDims:output:0*
T0*
_output_shapes
:m
#freq_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%freq_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%freq_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
freq_diff_words/strided_sliceStridedSlicefreq_diff_words/Shape:output:0,freq_diff_words/strided_slice/stack:output:0.freq_diff_words/strided_slice/stack_1:output:0.freq_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
freq_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
freq_diff_words/Reshape/shapePack&freq_diff_words/strided_slice:output:0(freq_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
freq_diff_words/ReshapeReshape#freq_diff_words/ExpandDims:output:0&freq_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
freq_of_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_adj/ExpandDims
ExpandDimsfeatures_10#freq_of_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
freq_of_adj/ShapeShapefreq_of_adj/ExpandDims:output:0*
T0*
_output_shapes
:i
freq_of_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!freq_of_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!freq_of_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_adj/strided_sliceStridedSlicefreq_of_adj/Shape:output:0(freq_of_adj/strided_slice/stack:output:0*freq_of_adj/strided_slice/stack_1:output:0*freq_of_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
freq_of_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_adj/Reshape/shapePack"freq_of_adj/strided_slice:output:0$freq_of_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_adj/ReshapeReshapefreq_of_adj/ExpandDims:output:0"freq_of_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
freq_of_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_adv/ExpandDims
ExpandDimsfeatures_11#freq_of_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
freq_of_adv/ShapeShapefreq_of_adv/ExpandDims:output:0*
T0*
_output_shapes
:i
freq_of_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!freq_of_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!freq_of_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_adv/strided_sliceStridedSlicefreq_of_adv/Shape:output:0(freq_of_adv/strided_slice/stack:output:0*freq_of_adv/strided_slice/stack_1:output:0*freq_of_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
freq_of_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_adv/Reshape/shapePack"freq_of_adv/strided_slice:output:0$freq_of_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_adv/ReshapeReshapefreq_of_adv/ExpandDims:output:0"freq_of_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#freq_of_distinct_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_distinct_adj/ExpandDims
ExpandDimsfeatures_12,freq_of_distinct_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
freq_of_distinct_adj/ShapeShape(freq_of_distinct_adj/ExpandDims:output:0*
T0*
_output_shapes
:r
(freq_of_distinct_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*freq_of_distinct_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*freq_of_distinct_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"freq_of_distinct_adj/strided_sliceStridedSlice#freq_of_distinct_adj/Shape:output:01freq_of_distinct_adj/strided_slice/stack:output:03freq_of_distinct_adj/strided_slice/stack_1:output:03freq_of_distinct_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$freq_of_distinct_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"freq_of_distinct_adj/Reshape/shapePack+freq_of_distinct_adj/strided_slice:output:0-freq_of_distinct_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
freq_of_distinct_adj/ReshapeReshape(freq_of_distinct_adj/ExpandDims:output:0+freq_of_distinct_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#freq_of_distinct_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_distinct_adv/ExpandDims
ExpandDimsfeatures_13,freq_of_distinct_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
freq_of_distinct_adv/ShapeShape(freq_of_distinct_adv/ExpandDims:output:0*
T0*
_output_shapes
:r
(freq_of_distinct_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*freq_of_distinct_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*freq_of_distinct_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"freq_of_distinct_adv/strided_sliceStridedSlice#freq_of_distinct_adv/Shape:output:01freq_of_distinct_adv/strided_slice/stack:output:03freq_of_distinct_adv/strided_slice/stack_1:output:03freq_of_distinct_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$freq_of_distinct_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"freq_of_distinct_adv/Reshape/shapePack+freq_of_distinct_adv/strided_slice:output:0-freq_of_distinct_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
freq_of_distinct_adv/ReshapeReshape(freq_of_distinct_adv/ExpandDims:output:0+freq_of_distinct_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
freq_of_noun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_noun/ExpandDims
ExpandDimsfeatures_14$freq_of_noun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
freq_of_noun/ShapeShape freq_of_noun/ExpandDims:output:0*
T0*
_output_shapes
:j
 freq_of_noun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"freq_of_noun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"freq_of_noun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_noun/strided_sliceStridedSlicefreq_of_noun/Shape:output:0)freq_of_noun/strided_slice/stack:output:0+freq_of_noun/strided_slice/stack_1:output:0+freq_of_noun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
freq_of_noun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_noun/Reshape/shapePack#freq_of_noun/strided_slice:output:0%freq_of_noun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_noun/ReshapeReshape freq_of_noun/ExpandDims:output:0#freq_of_noun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
freq_of_pronoun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_pronoun/ExpandDims
ExpandDimsfeatures_15'freq_of_pronoun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
freq_of_pronoun/ShapeShape#freq_of_pronoun/ExpandDims:output:0*
T0*
_output_shapes
:m
#freq_of_pronoun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%freq_of_pronoun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%freq_of_pronoun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
freq_of_pronoun/strided_sliceStridedSlicefreq_of_pronoun/Shape:output:0,freq_of_pronoun/strided_slice/stack:output:0.freq_of_pronoun/strided_slice/stack_1:output:0.freq_of_pronoun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
freq_of_pronoun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
freq_of_pronoun/Reshape/shapePack&freq_of_pronoun/strided_slice:output:0(freq_of_pronoun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
freq_of_pronoun/ReshapeReshape#freq_of_pronoun/ExpandDims:output:0&freq_of_pronoun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!freq_of_transition/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_transition/ExpandDims
ExpandDimsfeatures_16*freq_of_transition/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
freq_of_transition/ShapeShape&freq_of_transition/ExpandDims:output:0*
T0*
_output_shapes
:p
&freq_of_transition/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(freq_of_transition/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(freq_of_transition/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 freq_of_transition/strided_sliceStridedSlice!freq_of_transition/Shape:output:0/freq_of_transition/strided_slice/stack:output:01freq_of_transition/strided_slice/stack_1:output:01freq_of_transition/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"freq_of_transition/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 freq_of_transition/Reshape/shapePack)freq_of_transition/strided_slice:output:0+freq_of_transition/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
freq_of_transition/ReshapeReshape&freq_of_transition/ExpandDims:output:0)freq_of_transition/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
freq_of_verb/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_verb/ExpandDims
ExpandDimsfeatures_17$freq_of_verb/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
freq_of_verb/ShapeShape freq_of_verb/ExpandDims:output:0*
T0*
_output_shapes
:j
 freq_of_verb/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"freq_of_verb/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"freq_of_verb/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_verb/strided_sliceStridedSlicefreq_of_verb/Shape:output:0)freq_of_verb/strided_slice/stack:output:0+freq_of_verb/strided_slice/stack_1:output:0+freq_of_verb/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
freq_of_verb/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_verb/Reshape/shapePack#freq_of_verb/strided_slice:output:0%freq_of_verb/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_verb/ReshapeReshape freq_of_verb/ExpandDims:output:0#freq_of_verb/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"freq_of_wrong_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_wrong_words/ExpandDims
ExpandDimsfeatures_18+freq_of_wrong_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
freq_of_wrong_words/ShapeShape'freq_of_wrong_words/ExpandDims:output:0*
T0*
_output_shapes
:q
'freq_of_wrong_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)freq_of_wrong_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)freq_of_wrong_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!freq_of_wrong_words/strided_sliceStridedSlice"freq_of_wrong_words/Shape:output:00freq_of_wrong_words/strided_slice/stack:output:02freq_of_wrong_words/strided_slice/stack_1:output:02freq_of_wrong_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#freq_of_wrong_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!freq_of_wrong_words/Reshape/shapePack*freq_of_wrong_words/strided_slice:output:0,freq_of_wrong_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
freq_of_wrong_words/ReshapeReshape'freq_of_wrong_words/ExpandDims:output:0*freq_of_wrong_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#lexrank_avg_min_diff/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
lexrank_avg_min_diff/ExpandDims
ExpandDimsfeatures_19,lexrank_avg_min_diff/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
lexrank_avg_min_diff/ShapeShape(lexrank_avg_min_diff/ExpandDims:output:0*
T0*
_output_shapes
:r
(lexrank_avg_min_diff/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*lexrank_avg_min_diff/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*lexrank_avg_min_diff/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"lexrank_avg_min_diff/strided_sliceStridedSlice#lexrank_avg_min_diff/Shape:output:01lexrank_avg_min_diff/strided_slice/stack:output:03lexrank_avg_min_diff/strided_slice/stack_1:output:03lexrank_avg_min_diff/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$lexrank_avg_min_diff/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"lexrank_avg_min_diff/Reshape/shapePack+lexrank_avg_min_diff/strided_slice:output:0-lexrank_avg_min_diff/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
lexrank_avg_min_diff/ReshapeReshape(lexrank_avg_min_diff/ExpandDims:output:0+lexrank_avg_min_diff/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙o
$lexrank_interquartile/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
 lexrank_interquartile/ExpandDims
ExpandDimsfeatures_20-lexrank_interquartile/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lexrank_interquartile/ShapeShape)lexrank_interquartile/ExpandDims:output:0*
T0*
_output_shapes
:s
)lexrank_interquartile/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+lexrank_interquartile/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+lexrank_interquartile/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#lexrank_interquartile/strided_sliceStridedSlice$lexrank_interquartile/Shape:output:02lexrank_interquartile/strided_slice/stack:output:04lexrank_interquartile/strided_slice/stack_1:output:04lexrank_interquartile/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%lexrank_interquartile/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ˇ
#lexrank_interquartile/Reshape/shapePack,lexrank_interquartile/strided_slice:output:0.lexrank_interquartile/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ł
lexrank_interquartile/ReshapeReshape)lexrank_interquartile/ExpandDims:output:0,lexrank_interquartile/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
mcalpine_eflaw/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
mcalpine_eflaw/ExpandDims
ExpandDimsfeatures_21&mcalpine_eflaw/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
mcalpine_eflaw/ShapeShape"mcalpine_eflaw/ExpandDims:output:0*
T0*
_output_shapes
:l
"mcalpine_eflaw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$mcalpine_eflaw/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$mcalpine_eflaw/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
mcalpine_eflaw/strided_sliceStridedSlicemcalpine_eflaw/Shape:output:0+mcalpine_eflaw/strided_slice/stack:output:0-mcalpine_eflaw/strided_slice/stack_1:output:0-mcalpine_eflaw/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
mcalpine_eflaw/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :˘
mcalpine_eflaw/Reshape/shapePack%mcalpine_eflaw/strided_slice:output:0'mcalpine_eflaw/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
mcalpine_eflaw/ReshapeReshape"mcalpine_eflaw/ExpandDims:output:0%mcalpine_eflaw/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
noun_to_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
noun_to_adj/ExpandDims
ExpandDimsfeatures_22#noun_to_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
noun_to_adj/ShapeShapenoun_to_adj/ExpandDims:output:0*
T0*
_output_shapes
:i
noun_to_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!noun_to_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!noun_to_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
noun_to_adj/strided_sliceStridedSlicenoun_to_adj/Shape:output:0(noun_to_adj/strided_slice/stack:output:0*noun_to_adj/strided_slice/stack_1:output:0*noun_to_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
noun_to_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
noun_to_adj/Reshape/shapePack"noun_to_adj/strided_slice:output:0$noun_to_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
noun_to_adj/ReshapeReshapenoun_to_adj/ExpandDims:output:0"noun_to_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙o
$num_of_grammar_errors/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
 num_of_grammar_errors/ExpandDims
ExpandDimsfeatures_23-num_of_grammar_errors/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
num_of_grammar_errors/CastCast)num_of_grammar_errors/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
num_of_grammar_errors/ShapeShapenum_of_grammar_errors/Cast:y:0*
T0*
_output_shapes
:s
)num_of_grammar_errors/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+num_of_grammar_errors/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+num_of_grammar_errors/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#num_of_grammar_errors/strided_sliceStridedSlice$num_of_grammar_errors/Shape:output:02num_of_grammar_errors/strided_slice/stack:output:04num_of_grammar_errors/strided_slice/stack_1:output:04num_of_grammar_errors/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%num_of_grammar_errors/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ˇ
#num_of_grammar_errors/Reshape/shapePack,num_of_grammar_errors/strided_slice:output:0.num_of_grammar_errors/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:¨
num_of_grammar_errors/ReshapeReshapenum_of_grammar_errors/Cast:y:0,num_of_grammar_errors/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!num_of_short_forms/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
num_of_short_forms/ExpandDims
ExpandDimsfeatures_24*num_of_short_forms/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
num_of_short_forms/CastCast&num_of_short_forms/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙c
num_of_short_forms/ShapeShapenum_of_short_forms/Cast:y:0*
T0*
_output_shapes
:p
&num_of_short_forms/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(num_of_short_forms/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(num_of_short_forms/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 num_of_short_forms/strided_sliceStridedSlice!num_of_short_forms/Shape:output:0/num_of_short_forms/strided_slice/stack:output:01num_of_short_forms/strided_slice/stack_1:output:01num_of_short_forms/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"num_of_short_forms/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 num_of_short_forms/Reshape/shapePack)num_of_short_forms/strided_slice:output:0+num_of_short_forms/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
num_of_short_forms/ReshapeReshapenum_of_short_forms/Cast:y:0)num_of_short_forms/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#number_of_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
number_of_diff_words/ExpandDims
ExpandDimsfeatures_25,number_of_diff_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
number_of_diff_words/CastCast(number_of_diff_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
number_of_diff_words/ShapeShapenumber_of_diff_words/Cast:y:0*
T0*
_output_shapes
:r
(number_of_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*number_of_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*number_of_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"number_of_diff_words/strided_sliceStridedSlice#number_of_diff_words/Shape:output:01number_of_diff_words/strided_slice/stack:output:03number_of_diff_words/strided_slice/stack_1:output:03number_of_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$number_of_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"number_of_diff_words/Reshape/shapePack+number_of_diff_words/strided_slice:output:0-number_of_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ľ
number_of_diff_words/ReshapeReshapenumber_of_diff_words/Cast:y:0+number_of_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
number_of_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
number_of_words/ExpandDims
ExpandDimsfeatures_26'number_of_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
number_of_words/CastCast#number_of_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙]
number_of_words/ShapeShapenumber_of_words/Cast:y:0*
T0*
_output_shapes
:m
#number_of_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%number_of_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%number_of_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
number_of_words/strided_sliceStridedSlicenumber_of_words/Shape:output:0,number_of_words/strided_slice/stack:output:0.number_of_words/strided_slice/stack_1:output:0.number_of_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
number_of_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
number_of_words/Reshape/shapePack&number_of_words/strided_slice:output:0(number_of_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
number_of_words/ReshapeReshapenumber_of_words/Cast:y:0&number_of_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙j
phrase_diversity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
phrase_diversity/ExpandDims
ExpandDimsfeatures_27(phrase_diversity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙j
phrase_diversity/ShapeShape$phrase_diversity/ExpandDims:output:0*
T0*
_output_shapes
:n
$phrase_diversity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&phrase_diversity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&phrase_diversity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ś
phrase_diversity/strided_sliceStridedSlicephrase_diversity/Shape:output:0-phrase_diversity/strided_slice/stack:output:0/phrase_diversity/strided_slice/stack_1:output:0/phrase_diversity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 phrase_diversity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :¨
phrase_diversity/Reshape/shapePack'phrase_diversity/strided_slice:output:0)phrase_diversity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:¤
phrase_diversity/ReshapeReshape$phrase_diversity/ExpandDims:output:0'phrase_diversity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
punctuations/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
punctuations/ExpandDims
ExpandDimsfeatures_28$punctuations/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
punctuations/ShapeShape punctuations/ExpandDims:output:0*
T0*
_output_shapes
:j
 punctuations/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"punctuations/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"punctuations/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
punctuations/strided_sliceStridedSlicepunctuations/Shape:output:0)punctuations/strided_slice/stack:output:0+punctuations/strided_slice/stack_1:output:0+punctuations/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
punctuations/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
punctuations/Reshape/shapePack#punctuations/strided_slice:output:0%punctuations/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
punctuations/ReshapeReshape punctuations/ExpandDims:output:0#punctuations/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"sentence_complexity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
sentence_complexity/ExpandDims
ExpandDimsfeatures_29+sentence_complexity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
sentence_complexity/ShapeShape'sentence_complexity/ExpandDims:output:0*
T0*
_output_shapes
:q
'sentence_complexity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sentence_complexity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sentence_complexity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!sentence_complexity/strided_sliceStridedSlice"sentence_complexity/Shape:output:00sentence_complexity/strided_slice/stack:output:02sentence_complexity/strided_slice/stack_1:output:02sentence_complexity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sentence_complexity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!sentence_complexity/Reshape/shapePack*sentence_complexity/strided_slice:output:0,sentence_complexity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
sentence_complexity/ReshapeReshape'sentence_complexity/ExpandDims:output:0*sentence_complexity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!sentiment_compound/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
sentiment_compound/ExpandDims
ExpandDimsfeatures_30*sentiment_compound/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
sentiment_compound/ShapeShape&sentiment_compound/ExpandDims:output:0*
T0*
_output_shapes
:p
&sentiment_compound/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sentiment_compound/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sentiment_compound/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sentiment_compound/strided_sliceStridedSlice!sentiment_compound/Shape:output:0/sentiment_compound/strided_slice/stack:output:01sentiment_compound/strided_slice/stack_1:output:01sentiment_compound/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sentiment_compound/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 sentiment_compound/Reshape/shapePack)sentiment_compound/strided_slice:output:0+sentiment_compound/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
sentiment_compound/ReshapeReshape&sentiment_compound/ExpandDims:output:0)sentiment_compound/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!sentiment_negative/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
sentiment_negative/ExpandDims
ExpandDimsfeatures_31*sentiment_negative/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
sentiment_negative/ShapeShape&sentiment_negative/ExpandDims:output:0*
T0*
_output_shapes
:p
&sentiment_negative/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sentiment_negative/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sentiment_negative/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sentiment_negative/strided_sliceStridedSlice!sentiment_negative/Shape:output:0/sentiment_negative/strided_slice/stack:output:01sentiment_negative/strided_slice/stack_1:output:01sentiment_negative/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sentiment_negative/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 sentiment_negative/Reshape/shapePack)sentiment_negative/strided_slice:output:0+sentiment_negative/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
sentiment_negative/ReshapeReshape&sentiment_negative/ExpandDims:output:0)sentiment_negative/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!sentiment_positive/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
sentiment_positive/ExpandDims
ExpandDimsfeatures_32*sentiment_positive/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
sentiment_positive/ShapeShape&sentiment_positive/ExpandDims:output:0*
T0*
_output_shapes
:p
&sentiment_positive/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sentiment_positive/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sentiment_positive/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sentiment_positive/strided_sliceStridedSlice!sentiment_positive/Shape:output:0/sentiment_positive/strided_slice/stack:output:01sentiment_positive/strided_slice/stack_1:output:01sentiment_positive/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sentiment_positive/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 sentiment_positive/Reshape/shapePack)sentiment_positive/strided_slice:output:0+sentiment_positive/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
sentiment_positive/ReshapeReshape&sentiment_positive/ExpandDims:output:0)sentiment_positive/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"stopwords_frequency/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
stopwords_frequency/ExpandDims
ExpandDimsfeatures_33+stopwords_frequency/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
stopwords_frequency/ShapeShape'stopwords_frequency/ExpandDims:output:0*
T0*
_output_shapes
:q
'stopwords_frequency/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)stopwords_frequency/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)stopwords_frequency/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!stopwords_frequency/strided_sliceStridedSlice"stopwords_frequency/Shape:output:00stopwords_frequency/strided_slice/stack:output:02stopwords_frequency/strided_slice/stack_1:output:02stopwords_frequency/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#stopwords_frequency/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!stopwords_frequency/Reshape/shapePack*stopwords_frequency/strided_slice:output:0,stopwords_frequency/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
stopwords_frequency/ReshapeReshape'stopwords_frequency/ExpandDims:output:0*stopwords_frequency/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
text_standard/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
text_standard/ExpandDims
ExpandDimsfeatures_34%text_standard/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d
text_standard/ShapeShape!text_standard/ExpandDims:output:0*
T0*
_output_shapes
:k
!text_standard/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#text_standard/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#text_standard/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
text_standard/strided_sliceStridedSlicetext_standard/Shape:output:0*text_standard/strided_slice/stack:output:0,text_standard/strided_slice/stack_1:output:0,text_standard/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
text_standard/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
text_standard/Reshape/shapePack$text_standard/strided_slice:output:0&text_standard/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
text_standard/ReshapeReshape!text_standard/ExpandDims:output:0$text_standard/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙]
ttr/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙x
ttr/ExpandDims
ExpandDimsfeatures_35ttr/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P
	ttr/ShapeShapettr/ExpandDims:output:0*
T0*
_output_shapes
:a
ttr/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
ttr/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
ttr/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
ttr/strided_sliceStridedSlicettr/Shape:output:0 ttr/strided_slice/stack:output:0"ttr/strided_slice/stack_1:output:0"ttr/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
ttr/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ttr/Reshape/shapePackttr/strided_slice:output:0ttr/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:}
ttr/ReshapeReshapettr/ExpandDims:output:0ttr/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
verb_to_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
verb_to_adv/ExpandDims
ExpandDimsfeatures_36#verb_to_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
verb_to_adv/ShapeShapeverb_to_adv/ExpandDims:output:0*
T0*
_output_shapes
:i
verb_to_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!verb_to_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!verb_to_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
verb_to_adv/strided_sliceStridedSliceverb_to_adv/Shape:output:0(verb_to_adv/strided_slice/stack:output:0*verb_to_adv/strided_slice/stack_1:output:0*verb_to_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
verb_to_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
verb_to_adv/Reshape/shapePack"verb_to_adv/strided_slice:output:0$verb_to_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
verb_to_adv/ReshapeReshapeverb_to_adv/ExpandDims:output:0"verb_to_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź

concatConcatV2ARI/Reshape:output:0%Incorrect_form_ratio/Reshape:output:0 av_word_per_sen/Reshape:output:0 coherence_score/Reshape:output:0-dale_chall_readability_score/Reshape:output:0%flesch_kincaid_grade/Reshape:output:0$flesch_reading_ease/Reshape:output:0 freq_diff_words/Reshape:output:0freq_of_adj/Reshape:output:0freq_of_adv/Reshape:output:0%freq_of_distinct_adj/Reshape:output:0%freq_of_distinct_adv/Reshape:output:0freq_of_noun/Reshape:output:0 freq_of_pronoun/Reshape:output:0#freq_of_transition/Reshape:output:0freq_of_verb/Reshape:output:0$freq_of_wrong_words/Reshape:output:0%lexrank_avg_min_diff/Reshape:output:0&lexrank_interquartile/Reshape:output:0mcalpine_eflaw/Reshape:output:0noun_to_adj/Reshape:output:0&num_of_grammar_errors/Reshape:output:0#num_of_short_forms/Reshape:output:0%number_of_diff_words/Reshape:output:0 number_of_words/Reshape:output:0!phrase_diversity/Reshape:output:0punctuations/Reshape:output:0$sentence_complexity/Reshape:output:0#sentiment_compound/Reshape:output:0#sentiment_negative/Reshape:output:0#sentiment_positive/Reshape:output:0$stopwords_frequency/Reshape:output:0text_standard/Reshape:output:0ttr/Reshape:output:0verb_to_adv/Reshape:output:0concat/axis:output:0*
N#*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙#W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙#"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ŕ
_input_shapesŽ
Ť:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:M I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M	I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M
I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M!I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M"I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M#I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M$I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features
ž

'__inference_Hidden1_layer_call_fn_54297

inputs
unknown:
	unknown_0:
identity˘StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Hidden1_layer_call_and_return_conditional_losses_51516o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ÇA
Ť
__inference__traced_save_54473
file_prefix:
6savev2_sequential_2_hidden0_kernel_read_readvariableop8
4savev2_sequential_2_hidden0_bias_read_readvariableop:
6savev2_sequential_2_hidden1_kernel_read_readvariableop8
4savev2_sequential_2_hidden1_bias_read_readvariableop9
5savev2_sequential_2_output_kernel_read_readvariableop7
3savev2_sequential_2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopA
=savev2_adam_sequential_2_hidden0_kernel_m_read_readvariableop?
;savev2_adam_sequential_2_hidden0_bias_m_read_readvariableopA
=savev2_adam_sequential_2_hidden1_kernel_m_read_readvariableop?
;savev2_adam_sequential_2_hidden1_bias_m_read_readvariableop@
<savev2_adam_sequential_2_output_kernel_m_read_readvariableop>
:savev2_adam_sequential_2_output_bias_m_read_readvariableopA
=savev2_adam_sequential_2_hidden0_kernel_v_read_readvariableop?
;savev2_adam_sequential_2_hidden0_bias_v_read_readvariableopA
=savev2_adam_sequential_2_hidden1_kernel_v_read_readvariableop?
;savev2_adam_sequential_2_hidden1_bias_v_read_readvariableop@
<savev2_adam_sequential_2_output_kernel_v_read_readvariableop>
:savev2_adam_sequential_2_output_bias_v_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: á
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBýB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHŠ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_sequential_2_hidden0_kernel_read_readvariableop4savev2_sequential_2_hidden0_bias_read_readvariableop6savev2_sequential_2_hidden1_kernel_read_readvariableop4savev2_sequential_2_hidden1_bias_read_readvariableop5savev2_sequential_2_output_kernel_read_readvariableop3savev2_sequential_2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop=savev2_adam_sequential_2_hidden0_kernel_m_read_readvariableop;savev2_adam_sequential_2_hidden0_bias_m_read_readvariableop=savev2_adam_sequential_2_hidden1_kernel_m_read_readvariableop;savev2_adam_sequential_2_hidden1_bias_m_read_readvariableop<savev2_adam_sequential_2_output_kernel_m_read_readvariableop:savev2_adam_sequential_2_output_bias_m_read_readvariableop=savev2_adam_sequential_2_hidden0_kernel_v_read_readvariableop;savev2_adam_sequential_2_hidden0_bias_v_read_readvariableop=savev2_adam_sequential_2_hidden1_kernel_v_read_readvariableop;savev2_adam_sequential_2_hidden1_bias_v_read_readvariableop<savev2_adam_sequential_2_output_kernel_v_read_readvariableop:savev2_adam_sequential_2_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*ż
_input_shapes­
Ş: :#:::::: : : : : : : : : : : :#::::::#:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:#: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:#: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:#: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
ž

'__inference_Hidden0_layer_call_fn_54277

inputs
unknown:#
	unknown_0:
identity˘StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Hidden0_layer_call_and_return_conditional_losses_51499o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙#: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙#
 
_user_specified_nameinputs
ť
Â
I__inference_dense_features_layer_call_and_return_conditional_losses_51486
features	

features_1

features_2

features_3

features_4

features_5

features_6

features_7

features_8

features_9
features_10
features_11
features_12
features_13
features_14
features_15
features_16
features_17
features_18
features_19
features_20
features_21
features_22
features_23	
features_24	
features_25	
features_26	
features_27
features_28
features_29
features_30
features_31
features_32
features_33
features_34
features_35
features_36
identity]
ARI/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙u
ARI/ExpandDims
ExpandDimsfeaturesARI/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ARI/CastCastARI/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙E
	ARI/ShapeShapeARI/Cast:y:0*
T0*
_output_shapes
:a
ARI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
ARI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
ARI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
ARI/strided_sliceStridedSliceARI/Shape:output:0 ARI/strided_slice/stack:output:0"ARI/strided_slice/stack_1:output:0"ARI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
ARI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ARI/Reshape/shapePackARI/strided_slice:output:0ARI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
ARI/ReshapeReshapeARI/Cast:y:0ARI/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#Incorrect_form_ratio/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
Incorrect_form_ratio/ExpandDims
ExpandDims
features_1,Incorrect_form_ratio/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
Incorrect_form_ratio/ShapeShape(Incorrect_form_ratio/ExpandDims:output:0*
T0*
_output_shapes
:r
(Incorrect_form_ratio/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Incorrect_form_ratio/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Incorrect_form_ratio/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"Incorrect_form_ratio/strided_sliceStridedSlice#Incorrect_form_ratio/Shape:output:01Incorrect_form_ratio/strided_slice/stack:output:03Incorrect_form_ratio/strided_slice/stack_1:output:03Incorrect_form_ratio/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Incorrect_form_ratio/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"Incorrect_form_ratio/Reshape/shapePack+Incorrect_form_ratio/strided_slice:output:0-Incorrect_form_ratio/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
Incorrect_form_ratio/ReshapeReshape(Incorrect_form_ratio/ExpandDims:output:0+Incorrect_form_ratio/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
av_word_per_sen/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
av_word_per_sen/ExpandDims
ExpandDims
features_2'av_word_per_sen/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
av_word_per_sen/ShapeShape#av_word_per_sen/ExpandDims:output:0*
T0*
_output_shapes
:m
#av_word_per_sen/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%av_word_per_sen/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%av_word_per_sen/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
av_word_per_sen/strided_sliceStridedSliceav_word_per_sen/Shape:output:0,av_word_per_sen/strided_slice/stack:output:0.av_word_per_sen/strided_slice/stack_1:output:0.av_word_per_sen/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
av_word_per_sen/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
av_word_per_sen/Reshape/shapePack&av_word_per_sen/strided_slice:output:0(av_word_per_sen/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
av_word_per_sen/ReshapeReshape#av_word_per_sen/ExpandDims:output:0&av_word_per_sen/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
coherence_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
coherence_score/ExpandDims
ExpandDims
features_3'coherence_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
coherence_score/ShapeShape#coherence_score/ExpandDims:output:0*
T0*
_output_shapes
:m
#coherence_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%coherence_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%coherence_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
coherence_score/strided_sliceStridedSlicecoherence_score/Shape:output:0,coherence_score/strided_slice/stack:output:0.coherence_score/strided_slice/stack_1:output:0.coherence_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
coherence_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
coherence_score/Reshape/shapePack&coherence_score/strided_slice:output:0(coherence_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
coherence_score/ReshapeReshape#coherence_score/ExpandDims:output:0&coherence_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙v
+dale_chall_readability_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Š
'dale_chall_readability_score/ExpandDims
ExpandDims
features_64dale_chall_readability_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"dale_chall_readability_score/ShapeShape0dale_chall_readability_score/ExpandDims:output:0*
T0*
_output_shapes
:z
0dale_chall_readability_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dale_chall_readability_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dale_chall_readability_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dale_chall_readability_score/strided_sliceStridedSlice+dale_chall_readability_score/Shape:output:09dale_chall_readability_score/strided_slice/stack:output:0;dale_chall_readability_score/strided_slice/stack_1:output:0;dale_chall_readability_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,dale_chall_readability_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ě
*dale_chall_readability_score/Reshape/shapePack3dale_chall_readability_score/strided_slice:output:05dale_chall_readability_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Č
$dale_chall_readability_score/ReshapeReshape0dale_chall_readability_score/ExpandDims:output:03dale_chall_readability_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#flesch_kincaid_grade/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
flesch_kincaid_grade/ExpandDims
ExpandDims
features_7,flesch_kincaid_grade/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
flesch_kincaid_grade/ShapeShape(flesch_kincaid_grade/ExpandDims:output:0*
T0*
_output_shapes
:r
(flesch_kincaid_grade/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*flesch_kincaid_grade/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*flesch_kincaid_grade/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"flesch_kincaid_grade/strided_sliceStridedSlice#flesch_kincaid_grade/Shape:output:01flesch_kincaid_grade/strided_slice/stack:output:03flesch_kincaid_grade/strided_slice/stack_1:output:03flesch_kincaid_grade/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$flesch_kincaid_grade/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"flesch_kincaid_grade/Reshape/shapePack+flesch_kincaid_grade/strided_slice:output:0-flesch_kincaid_grade/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
flesch_kincaid_grade/ReshapeReshape(flesch_kincaid_grade/ExpandDims:output:0+flesch_kincaid_grade/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"flesch_reading_ease/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
flesch_reading_ease/ExpandDims
ExpandDims
features_8+flesch_reading_ease/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
flesch_reading_ease/ShapeShape'flesch_reading_ease/ExpandDims:output:0*
T0*
_output_shapes
:q
'flesch_reading_ease/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)flesch_reading_ease/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)flesch_reading_ease/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!flesch_reading_ease/strided_sliceStridedSlice"flesch_reading_ease/Shape:output:00flesch_reading_ease/strided_slice/stack:output:02flesch_reading_ease/strided_slice/stack_1:output:02flesch_reading_ease/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#flesch_reading_ease/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!flesch_reading_ease/Reshape/shapePack*flesch_reading_ease/strided_slice:output:0,flesch_reading_ease/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
flesch_reading_ease/ReshapeReshape'flesch_reading_ease/ExpandDims:output:0*flesch_reading_ease/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
freq_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_diff_words/ExpandDims
ExpandDims
features_9'freq_diff_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
freq_diff_words/ShapeShape#freq_diff_words/ExpandDims:output:0*
T0*
_output_shapes
:m
#freq_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%freq_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%freq_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
freq_diff_words/strided_sliceStridedSlicefreq_diff_words/Shape:output:0,freq_diff_words/strided_slice/stack:output:0.freq_diff_words/strided_slice/stack_1:output:0.freq_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
freq_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
freq_diff_words/Reshape/shapePack&freq_diff_words/strided_slice:output:0(freq_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
freq_diff_words/ReshapeReshape#freq_diff_words/ExpandDims:output:0&freq_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
freq_of_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_adj/ExpandDims
ExpandDimsfeatures_10#freq_of_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
freq_of_adj/ShapeShapefreq_of_adj/ExpandDims:output:0*
T0*
_output_shapes
:i
freq_of_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!freq_of_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!freq_of_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_adj/strided_sliceStridedSlicefreq_of_adj/Shape:output:0(freq_of_adj/strided_slice/stack:output:0*freq_of_adj/strided_slice/stack_1:output:0*freq_of_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
freq_of_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_adj/Reshape/shapePack"freq_of_adj/strided_slice:output:0$freq_of_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_adj/ReshapeReshapefreq_of_adj/ExpandDims:output:0"freq_of_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
freq_of_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_adv/ExpandDims
ExpandDimsfeatures_11#freq_of_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
freq_of_adv/ShapeShapefreq_of_adv/ExpandDims:output:0*
T0*
_output_shapes
:i
freq_of_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!freq_of_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!freq_of_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_adv/strided_sliceStridedSlicefreq_of_adv/Shape:output:0(freq_of_adv/strided_slice/stack:output:0*freq_of_adv/strided_slice/stack_1:output:0*freq_of_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
freq_of_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_adv/Reshape/shapePack"freq_of_adv/strided_slice:output:0$freq_of_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_adv/ReshapeReshapefreq_of_adv/ExpandDims:output:0"freq_of_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#freq_of_distinct_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_distinct_adj/ExpandDims
ExpandDimsfeatures_12,freq_of_distinct_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
freq_of_distinct_adj/ShapeShape(freq_of_distinct_adj/ExpandDims:output:0*
T0*
_output_shapes
:r
(freq_of_distinct_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*freq_of_distinct_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*freq_of_distinct_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"freq_of_distinct_adj/strided_sliceStridedSlice#freq_of_distinct_adj/Shape:output:01freq_of_distinct_adj/strided_slice/stack:output:03freq_of_distinct_adj/strided_slice/stack_1:output:03freq_of_distinct_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$freq_of_distinct_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"freq_of_distinct_adj/Reshape/shapePack+freq_of_distinct_adj/strided_slice:output:0-freq_of_distinct_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
freq_of_distinct_adj/ReshapeReshape(freq_of_distinct_adj/ExpandDims:output:0+freq_of_distinct_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#freq_of_distinct_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_distinct_adv/ExpandDims
ExpandDimsfeatures_13,freq_of_distinct_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
freq_of_distinct_adv/ShapeShape(freq_of_distinct_adv/ExpandDims:output:0*
T0*
_output_shapes
:r
(freq_of_distinct_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*freq_of_distinct_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*freq_of_distinct_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"freq_of_distinct_adv/strided_sliceStridedSlice#freq_of_distinct_adv/Shape:output:01freq_of_distinct_adv/strided_slice/stack:output:03freq_of_distinct_adv/strided_slice/stack_1:output:03freq_of_distinct_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$freq_of_distinct_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"freq_of_distinct_adv/Reshape/shapePack+freq_of_distinct_adv/strided_slice:output:0-freq_of_distinct_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
freq_of_distinct_adv/ReshapeReshape(freq_of_distinct_adv/ExpandDims:output:0+freq_of_distinct_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
freq_of_noun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_noun/ExpandDims
ExpandDimsfeatures_14$freq_of_noun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
freq_of_noun/ShapeShape freq_of_noun/ExpandDims:output:0*
T0*
_output_shapes
:j
 freq_of_noun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"freq_of_noun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"freq_of_noun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_noun/strided_sliceStridedSlicefreq_of_noun/Shape:output:0)freq_of_noun/strided_slice/stack:output:0+freq_of_noun/strided_slice/stack_1:output:0+freq_of_noun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
freq_of_noun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_noun/Reshape/shapePack#freq_of_noun/strided_slice:output:0%freq_of_noun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_noun/ReshapeReshape freq_of_noun/ExpandDims:output:0#freq_of_noun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
freq_of_pronoun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_pronoun/ExpandDims
ExpandDimsfeatures_15'freq_of_pronoun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
freq_of_pronoun/ShapeShape#freq_of_pronoun/ExpandDims:output:0*
T0*
_output_shapes
:m
#freq_of_pronoun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%freq_of_pronoun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%freq_of_pronoun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
freq_of_pronoun/strided_sliceStridedSlicefreq_of_pronoun/Shape:output:0,freq_of_pronoun/strided_slice/stack:output:0.freq_of_pronoun/strided_slice/stack_1:output:0.freq_of_pronoun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
freq_of_pronoun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
freq_of_pronoun/Reshape/shapePack&freq_of_pronoun/strided_slice:output:0(freq_of_pronoun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
freq_of_pronoun/ReshapeReshape#freq_of_pronoun/ExpandDims:output:0&freq_of_pronoun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!freq_of_transition/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_transition/ExpandDims
ExpandDimsfeatures_16*freq_of_transition/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
freq_of_transition/ShapeShape&freq_of_transition/ExpandDims:output:0*
T0*
_output_shapes
:p
&freq_of_transition/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(freq_of_transition/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(freq_of_transition/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 freq_of_transition/strided_sliceStridedSlice!freq_of_transition/Shape:output:0/freq_of_transition/strided_slice/stack:output:01freq_of_transition/strided_slice/stack_1:output:01freq_of_transition/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"freq_of_transition/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 freq_of_transition/Reshape/shapePack)freq_of_transition/strided_slice:output:0+freq_of_transition/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
freq_of_transition/ReshapeReshape&freq_of_transition/ExpandDims:output:0)freq_of_transition/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
freq_of_verb/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_verb/ExpandDims
ExpandDimsfeatures_17$freq_of_verb/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
freq_of_verb/ShapeShape freq_of_verb/ExpandDims:output:0*
T0*
_output_shapes
:j
 freq_of_verb/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"freq_of_verb/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"freq_of_verb/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_verb/strided_sliceStridedSlicefreq_of_verb/Shape:output:0)freq_of_verb/strided_slice/stack:output:0+freq_of_verb/strided_slice/stack_1:output:0+freq_of_verb/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
freq_of_verb/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_verb/Reshape/shapePack#freq_of_verb/strided_slice:output:0%freq_of_verb/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_verb/ReshapeReshape freq_of_verb/ExpandDims:output:0#freq_of_verb/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"freq_of_wrong_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_wrong_words/ExpandDims
ExpandDimsfeatures_18+freq_of_wrong_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
freq_of_wrong_words/ShapeShape'freq_of_wrong_words/ExpandDims:output:0*
T0*
_output_shapes
:q
'freq_of_wrong_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)freq_of_wrong_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)freq_of_wrong_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!freq_of_wrong_words/strided_sliceStridedSlice"freq_of_wrong_words/Shape:output:00freq_of_wrong_words/strided_slice/stack:output:02freq_of_wrong_words/strided_slice/stack_1:output:02freq_of_wrong_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#freq_of_wrong_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!freq_of_wrong_words/Reshape/shapePack*freq_of_wrong_words/strided_slice:output:0,freq_of_wrong_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
freq_of_wrong_words/ReshapeReshape'freq_of_wrong_words/ExpandDims:output:0*freq_of_wrong_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#lexrank_avg_min_diff/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
lexrank_avg_min_diff/ExpandDims
ExpandDimsfeatures_19,lexrank_avg_min_diff/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
lexrank_avg_min_diff/ShapeShape(lexrank_avg_min_diff/ExpandDims:output:0*
T0*
_output_shapes
:r
(lexrank_avg_min_diff/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*lexrank_avg_min_diff/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*lexrank_avg_min_diff/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"lexrank_avg_min_diff/strided_sliceStridedSlice#lexrank_avg_min_diff/Shape:output:01lexrank_avg_min_diff/strided_slice/stack:output:03lexrank_avg_min_diff/strided_slice/stack_1:output:03lexrank_avg_min_diff/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$lexrank_avg_min_diff/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"lexrank_avg_min_diff/Reshape/shapePack+lexrank_avg_min_diff/strided_slice:output:0-lexrank_avg_min_diff/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
lexrank_avg_min_diff/ReshapeReshape(lexrank_avg_min_diff/ExpandDims:output:0+lexrank_avg_min_diff/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙o
$lexrank_interquartile/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
 lexrank_interquartile/ExpandDims
ExpandDimsfeatures_20-lexrank_interquartile/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lexrank_interquartile/ShapeShape)lexrank_interquartile/ExpandDims:output:0*
T0*
_output_shapes
:s
)lexrank_interquartile/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+lexrank_interquartile/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+lexrank_interquartile/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#lexrank_interquartile/strided_sliceStridedSlice$lexrank_interquartile/Shape:output:02lexrank_interquartile/strided_slice/stack:output:04lexrank_interquartile/strided_slice/stack_1:output:04lexrank_interquartile/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%lexrank_interquartile/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ˇ
#lexrank_interquartile/Reshape/shapePack,lexrank_interquartile/strided_slice:output:0.lexrank_interquartile/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ł
lexrank_interquartile/ReshapeReshape)lexrank_interquartile/ExpandDims:output:0,lexrank_interquartile/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
mcalpine_eflaw/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
mcalpine_eflaw/ExpandDims
ExpandDimsfeatures_21&mcalpine_eflaw/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
mcalpine_eflaw/ShapeShape"mcalpine_eflaw/ExpandDims:output:0*
T0*
_output_shapes
:l
"mcalpine_eflaw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$mcalpine_eflaw/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$mcalpine_eflaw/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
mcalpine_eflaw/strided_sliceStridedSlicemcalpine_eflaw/Shape:output:0+mcalpine_eflaw/strided_slice/stack:output:0-mcalpine_eflaw/strided_slice/stack_1:output:0-mcalpine_eflaw/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
mcalpine_eflaw/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :˘
mcalpine_eflaw/Reshape/shapePack%mcalpine_eflaw/strided_slice:output:0'mcalpine_eflaw/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
mcalpine_eflaw/ReshapeReshape"mcalpine_eflaw/ExpandDims:output:0%mcalpine_eflaw/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
noun_to_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
noun_to_adj/ExpandDims
ExpandDimsfeatures_22#noun_to_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
noun_to_adj/ShapeShapenoun_to_adj/ExpandDims:output:0*
T0*
_output_shapes
:i
noun_to_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!noun_to_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!noun_to_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
noun_to_adj/strided_sliceStridedSlicenoun_to_adj/Shape:output:0(noun_to_adj/strided_slice/stack:output:0*noun_to_adj/strided_slice/stack_1:output:0*noun_to_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
noun_to_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
noun_to_adj/Reshape/shapePack"noun_to_adj/strided_slice:output:0$noun_to_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
noun_to_adj/ReshapeReshapenoun_to_adj/ExpandDims:output:0"noun_to_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙o
$num_of_grammar_errors/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
 num_of_grammar_errors/ExpandDims
ExpandDimsfeatures_23-num_of_grammar_errors/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
num_of_grammar_errors/CastCast)num_of_grammar_errors/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
num_of_grammar_errors/ShapeShapenum_of_grammar_errors/Cast:y:0*
T0*
_output_shapes
:s
)num_of_grammar_errors/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+num_of_grammar_errors/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+num_of_grammar_errors/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#num_of_grammar_errors/strided_sliceStridedSlice$num_of_grammar_errors/Shape:output:02num_of_grammar_errors/strided_slice/stack:output:04num_of_grammar_errors/strided_slice/stack_1:output:04num_of_grammar_errors/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%num_of_grammar_errors/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ˇ
#num_of_grammar_errors/Reshape/shapePack,num_of_grammar_errors/strided_slice:output:0.num_of_grammar_errors/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:¨
num_of_grammar_errors/ReshapeReshapenum_of_grammar_errors/Cast:y:0,num_of_grammar_errors/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!num_of_short_forms/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
num_of_short_forms/ExpandDims
ExpandDimsfeatures_24*num_of_short_forms/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
num_of_short_forms/CastCast&num_of_short_forms/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙c
num_of_short_forms/ShapeShapenum_of_short_forms/Cast:y:0*
T0*
_output_shapes
:p
&num_of_short_forms/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(num_of_short_forms/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(num_of_short_forms/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 num_of_short_forms/strided_sliceStridedSlice!num_of_short_forms/Shape:output:0/num_of_short_forms/strided_slice/stack:output:01num_of_short_forms/strided_slice/stack_1:output:01num_of_short_forms/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"num_of_short_forms/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 num_of_short_forms/Reshape/shapePack)num_of_short_forms/strided_slice:output:0+num_of_short_forms/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
num_of_short_forms/ReshapeReshapenum_of_short_forms/Cast:y:0)num_of_short_forms/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#number_of_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
number_of_diff_words/ExpandDims
ExpandDimsfeatures_25,number_of_diff_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
number_of_diff_words/CastCast(number_of_diff_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
number_of_diff_words/ShapeShapenumber_of_diff_words/Cast:y:0*
T0*
_output_shapes
:r
(number_of_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*number_of_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*number_of_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"number_of_diff_words/strided_sliceStridedSlice#number_of_diff_words/Shape:output:01number_of_diff_words/strided_slice/stack:output:03number_of_diff_words/strided_slice/stack_1:output:03number_of_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$number_of_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"number_of_diff_words/Reshape/shapePack+number_of_diff_words/strided_slice:output:0-number_of_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ľ
number_of_diff_words/ReshapeReshapenumber_of_diff_words/Cast:y:0+number_of_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
number_of_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
number_of_words/ExpandDims
ExpandDimsfeatures_26'number_of_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
number_of_words/CastCast#number_of_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙]
number_of_words/ShapeShapenumber_of_words/Cast:y:0*
T0*
_output_shapes
:m
#number_of_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%number_of_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%number_of_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
number_of_words/strided_sliceStridedSlicenumber_of_words/Shape:output:0,number_of_words/strided_slice/stack:output:0.number_of_words/strided_slice/stack_1:output:0.number_of_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
number_of_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
number_of_words/Reshape/shapePack&number_of_words/strided_slice:output:0(number_of_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
number_of_words/ReshapeReshapenumber_of_words/Cast:y:0&number_of_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙j
phrase_diversity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
phrase_diversity/ExpandDims
ExpandDimsfeatures_27(phrase_diversity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙j
phrase_diversity/ShapeShape$phrase_diversity/ExpandDims:output:0*
T0*
_output_shapes
:n
$phrase_diversity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&phrase_diversity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&phrase_diversity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ś
phrase_diversity/strided_sliceStridedSlicephrase_diversity/Shape:output:0-phrase_diversity/strided_slice/stack:output:0/phrase_diversity/strided_slice/stack_1:output:0/phrase_diversity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 phrase_diversity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :¨
phrase_diversity/Reshape/shapePack'phrase_diversity/strided_slice:output:0)phrase_diversity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:¤
phrase_diversity/ReshapeReshape$phrase_diversity/ExpandDims:output:0'phrase_diversity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
punctuations/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
punctuations/ExpandDims
ExpandDimsfeatures_28$punctuations/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
punctuations/ShapeShape punctuations/ExpandDims:output:0*
T0*
_output_shapes
:j
 punctuations/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"punctuations/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"punctuations/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
punctuations/strided_sliceStridedSlicepunctuations/Shape:output:0)punctuations/strided_slice/stack:output:0+punctuations/strided_slice/stack_1:output:0+punctuations/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
punctuations/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
punctuations/Reshape/shapePack#punctuations/strided_slice:output:0%punctuations/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
punctuations/ReshapeReshape punctuations/ExpandDims:output:0#punctuations/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"sentence_complexity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
sentence_complexity/ExpandDims
ExpandDimsfeatures_29+sentence_complexity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
sentence_complexity/ShapeShape'sentence_complexity/ExpandDims:output:0*
T0*
_output_shapes
:q
'sentence_complexity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sentence_complexity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sentence_complexity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!sentence_complexity/strided_sliceStridedSlice"sentence_complexity/Shape:output:00sentence_complexity/strided_slice/stack:output:02sentence_complexity/strided_slice/stack_1:output:02sentence_complexity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sentence_complexity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!sentence_complexity/Reshape/shapePack*sentence_complexity/strided_slice:output:0,sentence_complexity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
sentence_complexity/ReshapeReshape'sentence_complexity/ExpandDims:output:0*sentence_complexity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!sentiment_compound/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
sentiment_compound/ExpandDims
ExpandDimsfeatures_30*sentiment_compound/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
sentiment_compound/ShapeShape&sentiment_compound/ExpandDims:output:0*
T0*
_output_shapes
:p
&sentiment_compound/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sentiment_compound/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sentiment_compound/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sentiment_compound/strided_sliceStridedSlice!sentiment_compound/Shape:output:0/sentiment_compound/strided_slice/stack:output:01sentiment_compound/strided_slice/stack_1:output:01sentiment_compound/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sentiment_compound/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 sentiment_compound/Reshape/shapePack)sentiment_compound/strided_slice:output:0+sentiment_compound/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
sentiment_compound/ReshapeReshape&sentiment_compound/ExpandDims:output:0)sentiment_compound/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!sentiment_negative/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
sentiment_negative/ExpandDims
ExpandDimsfeatures_31*sentiment_negative/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
sentiment_negative/ShapeShape&sentiment_negative/ExpandDims:output:0*
T0*
_output_shapes
:p
&sentiment_negative/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sentiment_negative/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sentiment_negative/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sentiment_negative/strided_sliceStridedSlice!sentiment_negative/Shape:output:0/sentiment_negative/strided_slice/stack:output:01sentiment_negative/strided_slice/stack_1:output:01sentiment_negative/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sentiment_negative/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 sentiment_negative/Reshape/shapePack)sentiment_negative/strided_slice:output:0+sentiment_negative/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
sentiment_negative/ReshapeReshape&sentiment_negative/ExpandDims:output:0)sentiment_negative/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!sentiment_positive/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
sentiment_positive/ExpandDims
ExpandDimsfeatures_32*sentiment_positive/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
sentiment_positive/ShapeShape&sentiment_positive/ExpandDims:output:0*
T0*
_output_shapes
:p
&sentiment_positive/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sentiment_positive/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sentiment_positive/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sentiment_positive/strided_sliceStridedSlice!sentiment_positive/Shape:output:0/sentiment_positive/strided_slice/stack:output:01sentiment_positive/strided_slice/stack_1:output:01sentiment_positive/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sentiment_positive/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 sentiment_positive/Reshape/shapePack)sentiment_positive/strided_slice:output:0+sentiment_positive/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
sentiment_positive/ReshapeReshape&sentiment_positive/ExpandDims:output:0)sentiment_positive/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"stopwords_frequency/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
stopwords_frequency/ExpandDims
ExpandDimsfeatures_33+stopwords_frequency/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
stopwords_frequency/ShapeShape'stopwords_frequency/ExpandDims:output:0*
T0*
_output_shapes
:q
'stopwords_frequency/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)stopwords_frequency/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)stopwords_frequency/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!stopwords_frequency/strided_sliceStridedSlice"stopwords_frequency/Shape:output:00stopwords_frequency/strided_slice/stack:output:02stopwords_frequency/strided_slice/stack_1:output:02stopwords_frequency/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#stopwords_frequency/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!stopwords_frequency/Reshape/shapePack*stopwords_frequency/strided_slice:output:0,stopwords_frequency/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
stopwords_frequency/ReshapeReshape'stopwords_frequency/ExpandDims:output:0*stopwords_frequency/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
text_standard/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
text_standard/ExpandDims
ExpandDimsfeatures_34%text_standard/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d
text_standard/ShapeShape!text_standard/ExpandDims:output:0*
T0*
_output_shapes
:k
!text_standard/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#text_standard/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#text_standard/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
text_standard/strided_sliceStridedSlicetext_standard/Shape:output:0*text_standard/strided_slice/stack:output:0,text_standard/strided_slice/stack_1:output:0,text_standard/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
text_standard/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
text_standard/Reshape/shapePack$text_standard/strided_slice:output:0&text_standard/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
text_standard/ReshapeReshape!text_standard/ExpandDims:output:0$text_standard/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙]
ttr/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙x
ttr/ExpandDims
ExpandDimsfeatures_35ttr/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P
	ttr/ShapeShapettr/ExpandDims:output:0*
T0*
_output_shapes
:a
ttr/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
ttr/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
ttr/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
ttr/strided_sliceStridedSlicettr/Shape:output:0 ttr/strided_slice/stack:output:0"ttr/strided_slice/stack_1:output:0"ttr/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
ttr/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ttr/Reshape/shapePackttr/strided_slice:output:0ttr/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:}
ttr/ReshapeReshapettr/ExpandDims:output:0ttr/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
verb_to_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
verb_to_adv/ExpandDims
ExpandDimsfeatures_36#verb_to_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
verb_to_adv/ShapeShapeverb_to_adv/ExpandDims:output:0*
T0*
_output_shapes
:i
verb_to_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!verb_to_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!verb_to_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
verb_to_adv/strided_sliceStridedSliceverb_to_adv/Shape:output:0(verb_to_adv/strided_slice/stack:output:0*verb_to_adv/strided_slice/stack_1:output:0*verb_to_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
verb_to_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
verb_to_adv/Reshape/shapePack"verb_to_adv/strided_slice:output:0$verb_to_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
verb_to_adv/ReshapeReshapeverb_to_adv/ExpandDims:output:0"verb_to_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź

concatConcatV2ARI/Reshape:output:0%Incorrect_form_ratio/Reshape:output:0 av_word_per_sen/Reshape:output:0 coherence_score/Reshape:output:0-dale_chall_readability_score/Reshape:output:0%flesch_kincaid_grade/Reshape:output:0$flesch_reading_ease/Reshape:output:0 freq_diff_words/Reshape:output:0freq_of_adj/Reshape:output:0freq_of_adv/Reshape:output:0%freq_of_distinct_adj/Reshape:output:0%freq_of_distinct_adv/Reshape:output:0freq_of_noun/Reshape:output:0 freq_of_pronoun/Reshape:output:0#freq_of_transition/Reshape:output:0freq_of_verb/Reshape:output:0$freq_of_wrong_words/Reshape:output:0%lexrank_avg_min_diff/Reshape:output:0&lexrank_interquartile/Reshape:output:0mcalpine_eflaw/Reshape:output:0noun_to_adj/Reshape:output:0&num_of_grammar_errors/Reshape:output:0#num_of_short_forms/Reshape:output:0%number_of_diff_words/Reshape:output:0 number_of_words/Reshape:output:0!phrase_diversity/Reshape:output:0punctuations/Reshape:output:0$sentence_complexity/Reshape:output:0#sentiment_compound/Reshape:output:0#sentiment_negative/Reshape:output:0#sentiment_positive/Reshape:output:0$stopwords_frequency/Reshape:output:0text_standard/Reshape:output:0ttr/Reshape:output:0verb_to_adv/Reshape:output:0concat/axis:output:0*
N#*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙#W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙#"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ŕ
_input_shapesŽ
Ť:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:M I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M	I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M
I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M!I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M"I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M#I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features:M$I
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
features
Ä	
ň
A__inference_Output_layer_call_and_return_conditional_losses_54327

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
7
 

,__inference_sequential_2_layer_call_fn_52505

inputs_ari	
inputs_incorrect_form_ratio
inputs_av_word_per_sen
inputs_coherence_score
inputs_cohesion
inputs_corrected_text'
#inputs_dale_chall_readability_score
inputs_flesch_kincaid_grade
inputs_flesch_reading_ease
inputs_freq_diff_words
inputs_freq_of_adj
inputs_freq_of_adv
inputs_freq_of_distinct_adj
inputs_freq_of_distinct_adv
inputs_freq_of_noun
inputs_freq_of_pronoun
inputs_freq_of_transition
inputs_freq_of_verb
inputs_freq_of_wrong_words
inputs_lexrank_avg_min_diff 
inputs_lexrank_interquartile
inputs_mcalpine_eflaw
inputs_noun_to_adj 
inputs_num_of_grammar_errors	
inputs_num_of_short_forms	
inputs_number_of_diff_words	
inputs_number_of_words	
inputs_phrase_diversity
inputs_punctuations
inputs_sentence_complexity
inputs_sentiment_compound
inputs_sentiment_negative
inputs_sentiment_positive
inputs_stopwords_frequency
inputs_text_standard

inputs_ttr
inputs_verb_to_adv
unknown:#
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity˘StatefulPartitionedCallż

StatefulPartitionedCallStatefulPartitionedCall
inputs_ariinputs_incorrect_form_ratioinputs_av_word_per_seninputs_coherence_scoreinputs_cohesioninputs_corrected_text#inputs_dale_chall_readability_scoreinputs_flesch_kincaid_gradeinputs_flesch_reading_easeinputs_freq_diff_wordsinputs_freq_of_adjinputs_freq_of_advinputs_freq_of_distinct_adjinputs_freq_of_distinct_advinputs_freq_of_nouninputs_freq_of_pronouninputs_freq_of_transitioninputs_freq_of_verbinputs_freq_of_wrong_wordsinputs_lexrank_avg_min_diffinputs_lexrank_interquartileinputs_mcalpine_eflawinputs_noun_to_adjinputs_num_of_grammar_errorsinputs_num_of_short_formsinputs_number_of_diff_wordsinputs_number_of_wordsinputs_phrase_diversityinputs_punctuationsinputs_sentence_complexityinputs_sentiment_compoundinputs_sentiment_negativeinputs_sentiment_positiveinputs_stopwords_frequencyinputs_text_standard
inputs_ttrinputs_verb_to_advunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*6
Tin/
-2+					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

%&'()**-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_51539o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ě
_input_shapesş
ˇ:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs/ARI:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/Incorrect_form_ratio:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/av_word_per_sen:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/coherence_score:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs/cohesion:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs/corrected_text:hd
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
=
_user_specified_name%#inputs/dale_chall_readability_score:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/flesch_kincaid_grade:_[
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/flesch_reading_ease:[	W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/freq_diff_words:W
S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/freq_of_adj:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/freq_of_adv:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/freq_of_distinct_adj:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/freq_of_distinct_adv:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs/freq_of_noun:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/freq_of_pronoun:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/freq_of_transition:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs/freq_of_verb:_[
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/freq_of_wrong_words:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/lexrank_avg_min_diff:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_nameinputs/lexrank_interquartile:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs/mcalpine_eflaw:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/noun_to_adj:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_nameinputs/num_of_grammar_errors:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/num_of_short_forms:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/number_of_diff_words:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/number_of_words:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs/phrase_diversity:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs/punctuations:_[
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/sentence_complexity:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/sentiment_compound:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/sentiment_negative:^ Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/sentiment_positive:_![
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/stopwords_frequency:Y"U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs/text_standard:O#K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs/ttr:W$S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/verb_to_adv
Č
Ę	
I__inference_dense_features_layer_call_and_return_conditional_losses_54268
features_ari	!
features_incorrect_form_ratio
features_av_word_per_sen
features_coherence_score
features_cohesion
features_corrected_text)
%features_dale_chall_readability_score!
features_flesch_kincaid_grade 
features_flesch_reading_ease
features_freq_diff_words
features_freq_of_adj
features_freq_of_adv!
features_freq_of_distinct_adj!
features_freq_of_distinct_adv
features_freq_of_noun
features_freq_of_pronoun
features_freq_of_transition
features_freq_of_verb 
features_freq_of_wrong_words!
features_lexrank_avg_min_diff"
features_lexrank_interquartile
features_mcalpine_eflaw
features_noun_to_adj"
features_num_of_grammar_errors	
features_num_of_short_forms	!
features_number_of_diff_words	
features_number_of_words	
features_phrase_diversity
features_punctuations 
features_sentence_complexity
features_sentiment_compound
features_sentiment_negative
features_sentiment_positive 
features_stopwords_frequency
features_text_standard
features_ttr
features_verb_to_adv
identity]
ARI/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙y
ARI/ExpandDims
ExpandDimsfeatures_ariARI/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ARI/CastCastARI/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙E
	ARI/ShapeShapeARI/Cast:y:0*
T0*
_output_shapes
:a
ARI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
ARI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
ARI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
ARI/strided_sliceStridedSliceARI/Shape:output:0 ARI/strided_slice/stack:output:0"ARI/strided_slice/stack_1:output:0"ARI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
ARI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ARI/Reshape/shapePackARI/strided_slice:output:0ARI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
ARI/ReshapeReshapeARI/Cast:y:0ARI/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#Incorrect_form_ratio/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
Incorrect_form_ratio/ExpandDims
ExpandDimsfeatures_incorrect_form_ratio,Incorrect_form_ratio/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
Incorrect_form_ratio/ShapeShape(Incorrect_form_ratio/ExpandDims:output:0*
T0*
_output_shapes
:r
(Incorrect_form_ratio/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Incorrect_form_ratio/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Incorrect_form_ratio/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"Incorrect_form_ratio/strided_sliceStridedSlice#Incorrect_form_ratio/Shape:output:01Incorrect_form_ratio/strided_slice/stack:output:03Incorrect_form_ratio/strided_slice/stack_1:output:03Incorrect_form_ratio/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Incorrect_form_ratio/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"Incorrect_form_ratio/Reshape/shapePack+Incorrect_form_ratio/strided_slice:output:0-Incorrect_form_ratio/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
Incorrect_form_ratio/ReshapeReshape(Incorrect_form_ratio/ExpandDims:output:0+Incorrect_form_ratio/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
av_word_per_sen/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
av_word_per_sen/ExpandDims
ExpandDimsfeatures_av_word_per_sen'av_word_per_sen/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
av_word_per_sen/ShapeShape#av_word_per_sen/ExpandDims:output:0*
T0*
_output_shapes
:m
#av_word_per_sen/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%av_word_per_sen/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%av_word_per_sen/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
av_word_per_sen/strided_sliceStridedSliceav_word_per_sen/Shape:output:0,av_word_per_sen/strided_slice/stack:output:0.av_word_per_sen/strided_slice/stack_1:output:0.av_word_per_sen/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
av_word_per_sen/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
av_word_per_sen/Reshape/shapePack&av_word_per_sen/strided_slice:output:0(av_word_per_sen/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
av_word_per_sen/ReshapeReshape#av_word_per_sen/ExpandDims:output:0&av_word_per_sen/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
coherence_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
coherence_score/ExpandDims
ExpandDimsfeatures_coherence_score'coherence_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
coherence_score/ShapeShape#coherence_score/ExpandDims:output:0*
T0*
_output_shapes
:m
#coherence_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%coherence_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%coherence_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
coherence_score/strided_sliceStridedSlicecoherence_score/Shape:output:0,coherence_score/strided_slice/stack:output:0.coherence_score/strided_slice/stack_1:output:0.coherence_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
coherence_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
coherence_score/Reshape/shapePack&coherence_score/strided_slice:output:0(coherence_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
coherence_score/ReshapeReshape#coherence_score/ExpandDims:output:0&coherence_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙v
+dale_chall_readability_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ä
'dale_chall_readability_score/ExpandDims
ExpandDims%features_dale_chall_readability_score4dale_chall_readability_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"dale_chall_readability_score/ShapeShape0dale_chall_readability_score/ExpandDims:output:0*
T0*
_output_shapes
:z
0dale_chall_readability_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dale_chall_readability_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dale_chall_readability_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dale_chall_readability_score/strided_sliceStridedSlice+dale_chall_readability_score/Shape:output:09dale_chall_readability_score/strided_slice/stack:output:0;dale_chall_readability_score/strided_slice/stack_1:output:0;dale_chall_readability_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,dale_chall_readability_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ě
*dale_chall_readability_score/Reshape/shapePack3dale_chall_readability_score/strided_slice:output:05dale_chall_readability_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Č
$dale_chall_readability_score/ReshapeReshape0dale_chall_readability_score/ExpandDims:output:03dale_chall_readability_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#flesch_kincaid_grade/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
flesch_kincaid_grade/ExpandDims
ExpandDimsfeatures_flesch_kincaid_grade,flesch_kincaid_grade/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
flesch_kincaid_grade/ShapeShape(flesch_kincaid_grade/ExpandDims:output:0*
T0*
_output_shapes
:r
(flesch_kincaid_grade/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*flesch_kincaid_grade/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*flesch_kincaid_grade/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"flesch_kincaid_grade/strided_sliceStridedSlice#flesch_kincaid_grade/Shape:output:01flesch_kincaid_grade/strided_slice/stack:output:03flesch_kincaid_grade/strided_slice/stack_1:output:03flesch_kincaid_grade/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$flesch_kincaid_grade/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"flesch_kincaid_grade/Reshape/shapePack+flesch_kincaid_grade/strided_slice:output:0-flesch_kincaid_grade/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
flesch_kincaid_grade/ReshapeReshape(flesch_kincaid_grade/ExpandDims:output:0+flesch_kincaid_grade/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"flesch_reading_ease/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Š
flesch_reading_ease/ExpandDims
ExpandDimsfeatures_flesch_reading_ease+flesch_reading_ease/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
flesch_reading_ease/ShapeShape'flesch_reading_ease/ExpandDims:output:0*
T0*
_output_shapes
:q
'flesch_reading_ease/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)flesch_reading_ease/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)flesch_reading_ease/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!flesch_reading_ease/strided_sliceStridedSlice"flesch_reading_ease/Shape:output:00flesch_reading_ease/strided_slice/stack:output:02flesch_reading_ease/strided_slice/stack_1:output:02flesch_reading_ease/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#flesch_reading_ease/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!flesch_reading_ease/Reshape/shapePack*flesch_reading_ease/strided_slice:output:0,flesch_reading_ease/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
flesch_reading_ease/ReshapeReshape'flesch_reading_ease/ExpandDims:output:0*flesch_reading_ease/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
freq_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_diff_words/ExpandDims
ExpandDimsfeatures_freq_diff_words'freq_diff_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
freq_diff_words/ShapeShape#freq_diff_words/ExpandDims:output:0*
T0*
_output_shapes
:m
#freq_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%freq_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%freq_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
freq_diff_words/strided_sliceStridedSlicefreq_diff_words/Shape:output:0,freq_diff_words/strided_slice/stack:output:0.freq_diff_words/strided_slice/stack_1:output:0.freq_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
freq_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
freq_diff_words/Reshape/shapePack&freq_diff_words/strided_slice:output:0(freq_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
freq_diff_words/ReshapeReshape#freq_diff_words/ExpandDims:output:0&freq_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
freq_of_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_adj/ExpandDims
ExpandDimsfeatures_freq_of_adj#freq_of_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
freq_of_adj/ShapeShapefreq_of_adj/ExpandDims:output:0*
T0*
_output_shapes
:i
freq_of_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!freq_of_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!freq_of_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_adj/strided_sliceStridedSlicefreq_of_adj/Shape:output:0(freq_of_adj/strided_slice/stack:output:0*freq_of_adj/strided_slice/stack_1:output:0*freq_of_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
freq_of_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_adj/Reshape/shapePack"freq_of_adj/strided_slice:output:0$freq_of_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_adj/ReshapeReshapefreq_of_adj/ExpandDims:output:0"freq_of_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
freq_of_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_adv/ExpandDims
ExpandDimsfeatures_freq_of_adv#freq_of_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
freq_of_adv/ShapeShapefreq_of_adv/ExpandDims:output:0*
T0*
_output_shapes
:i
freq_of_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!freq_of_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!freq_of_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_adv/strided_sliceStridedSlicefreq_of_adv/Shape:output:0(freq_of_adv/strided_slice/stack:output:0*freq_of_adv/strided_slice/stack_1:output:0*freq_of_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
freq_of_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_adv/Reshape/shapePack"freq_of_adv/strided_slice:output:0$freq_of_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_adv/ReshapeReshapefreq_of_adv/ExpandDims:output:0"freq_of_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#freq_of_distinct_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
freq_of_distinct_adj/ExpandDims
ExpandDimsfeatures_freq_of_distinct_adj,freq_of_distinct_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
freq_of_distinct_adj/ShapeShape(freq_of_distinct_adj/ExpandDims:output:0*
T0*
_output_shapes
:r
(freq_of_distinct_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*freq_of_distinct_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*freq_of_distinct_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"freq_of_distinct_adj/strided_sliceStridedSlice#freq_of_distinct_adj/Shape:output:01freq_of_distinct_adj/strided_slice/stack:output:03freq_of_distinct_adj/strided_slice/stack_1:output:03freq_of_distinct_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$freq_of_distinct_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"freq_of_distinct_adj/Reshape/shapePack+freq_of_distinct_adj/strided_slice:output:0-freq_of_distinct_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
freq_of_distinct_adj/ReshapeReshape(freq_of_distinct_adj/ExpandDims:output:0+freq_of_distinct_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#freq_of_distinct_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
freq_of_distinct_adv/ExpandDims
ExpandDimsfeatures_freq_of_distinct_adv,freq_of_distinct_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
freq_of_distinct_adv/ShapeShape(freq_of_distinct_adv/ExpandDims:output:0*
T0*
_output_shapes
:r
(freq_of_distinct_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*freq_of_distinct_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*freq_of_distinct_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"freq_of_distinct_adv/strided_sliceStridedSlice#freq_of_distinct_adv/Shape:output:01freq_of_distinct_adv/strided_slice/stack:output:03freq_of_distinct_adv/strided_slice/stack_1:output:03freq_of_distinct_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$freq_of_distinct_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"freq_of_distinct_adv/Reshape/shapePack+freq_of_distinct_adv/strided_slice:output:0-freq_of_distinct_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
freq_of_distinct_adv/ReshapeReshape(freq_of_distinct_adv/ExpandDims:output:0+freq_of_distinct_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
freq_of_noun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_noun/ExpandDims
ExpandDimsfeatures_freq_of_noun$freq_of_noun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
freq_of_noun/ShapeShape freq_of_noun/ExpandDims:output:0*
T0*
_output_shapes
:j
 freq_of_noun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"freq_of_noun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"freq_of_noun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_noun/strided_sliceStridedSlicefreq_of_noun/Shape:output:0)freq_of_noun/strided_slice/stack:output:0+freq_of_noun/strided_slice/stack_1:output:0+freq_of_noun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
freq_of_noun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_noun/Reshape/shapePack#freq_of_noun/strided_slice:output:0%freq_of_noun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_noun/ReshapeReshape freq_of_noun/ExpandDims:output:0#freq_of_noun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
freq_of_pronoun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_pronoun/ExpandDims
ExpandDimsfeatures_freq_of_pronoun'freq_of_pronoun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
freq_of_pronoun/ShapeShape#freq_of_pronoun/ExpandDims:output:0*
T0*
_output_shapes
:m
#freq_of_pronoun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%freq_of_pronoun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%freq_of_pronoun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
freq_of_pronoun/strided_sliceStridedSlicefreq_of_pronoun/Shape:output:0,freq_of_pronoun/strided_slice/stack:output:0.freq_of_pronoun/strided_slice/stack_1:output:0.freq_of_pronoun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
freq_of_pronoun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
freq_of_pronoun/Reshape/shapePack&freq_of_pronoun/strided_slice:output:0(freq_of_pronoun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
freq_of_pronoun/ReshapeReshape#freq_of_pronoun/ExpandDims:output:0&freq_of_pronoun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!freq_of_transition/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ś
freq_of_transition/ExpandDims
ExpandDimsfeatures_freq_of_transition*freq_of_transition/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
freq_of_transition/ShapeShape&freq_of_transition/ExpandDims:output:0*
T0*
_output_shapes
:p
&freq_of_transition/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(freq_of_transition/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(freq_of_transition/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 freq_of_transition/strided_sliceStridedSlice!freq_of_transition/Shape:output:0/freq_of_transition/strided_slice/stack:output:01freq_of_transition/strided_slice/stack_1:output:01freq_of_transition/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"freq_of_transition/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 freq_of_transition/Reshape/shapePack)freq_of_transition/strided_slice:output:0+freq_of_transition/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
freq_of_transition/ReshapeReshape&freq_of_transition/ExpandDims:output:0)freq_of_transition/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
freq_of_verb/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_verb/ExpandDims
ExpandDimsfeatures_freq_of_verb$freq_of_verb/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
freq_of_verb/ShapeShape freq_of_verb/ExpandDims:output:0*
T0*
_output_shapes
:j
 freq_of_verb/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"freq_of_verb/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"freq_of_verb/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_verb/strided_sliceStridedSlicefreq_of_verb/Shape:output:0)freq_of_verb/strided_slice/stack:output:0+freq_of_verb/strided_slice/stack_1:output:0+freq_of_verb/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
freq_of_verb/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_verb/Reshape/shapePack#freq_of_verb/strided_slice:output:0%freq_of_verb/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_verb/ReshapeReshape freq_of_verb/ExpandDims:output:0#freq_of_verb/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"freq_of_wrong_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Š
freq_of_wrong_words/ExpandDims
ExpandDimsfeatures_freq_of_wrong_words+freq_of_wrong_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
freq_of_wrong_words/ShapeShape'freq_of_wrong_words/ExpandDims:output:0*
T0*
_output_shapes
:q
'freq_of_wrong_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)freq_of_wrong_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)freq_of_wrong_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!freq_of_wrong_words/strided_sliceStridedSlice"freq_of_wrong_words/Shape:output:00freq_of_wrong_words/strided_slice/stack:output:02freq_of_wrong_words/strided_slice/stack_1:output:02freq_of_wrong_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#freq_of_wrong_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!freq_of_wrong_words/Reshape/shapePack*freq_of_wrong_words/strided_slice:output:0,freq_of_wrong_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
freq_of_wrong_words/ReshapeReshape'freq_of_wrong_words/ExpandDims:output:0*freq_of_wrong_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#lexrank_avg_min_diff/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
lexrank_avg_min_diff/ExpandDims
ExpandDimsfeatures_lexrank_avg_min_diff,lexrank_avg_min_diff/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
lexrank_avg_min_diff/ShapeShape(lexrank_avg_min_diff/ExpandDims:output:0*
T0*
_output_shapes
:r
(lexrank_avg_min_diff/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*lexrank_avg_min_diff/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*lexrank_avg_min_diff/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"lexrank_avg_min_diff/strided_sliceStridedSlice#lexrank_avg_min_diff/Shape:output:01lexrank_avg_min_diff/strided_slice/stack:output:03lexrank_avg_min_diff/strided_slice/stack_1:output:03lexrank_avg_min_diff/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$lexrank_avg_min_diff/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"lexrank_avg_min_diff/Reshape/shapePack+lexrank_avg_min_diff/strided_slice:output:0-lexrank_avg_min_diff/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
lexrank_avg_min_diff/ReshapeReshape(lexrank_avg_min_diff/ExpandDims:output:0+lexrank_avg_min_diff/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙o
$lexrank_interquartile/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ż
 lexrank_interquartile/ExpandDims
ExpandDimsfeatures_lexrank_interquartile-lexrank_interquartile/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lexrank_interquartile/ShapeShape)lexrank_interquartile/ExpandDims:output:0*
T0*
_output_shapes
:s
)lexrank_interquartile/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+lexrank_interquartile/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+lexrank_interquartile/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#lexrank_interquartile/strided_sliceStridedSlice$lexrank_interquartile/Shape:output:02lexrank_interquartile/strided_slice/stack:output:04lexrank_interquartile/strided_slice/stack_1:output:04lexrank_interquartile/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%lexrank_interquartile/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ˇ
#lexrank_interquartile/Reshape/shapePack,lexrank_interquartile/strided_slice:output:0.lexrank_interquartile/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ł
lexrank_interquartile/ReshapeReshape)lexrank_interquartile/ExpandDims:output:0,lexrank_interquartile/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
mcalpine_eflaw/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
mcalpine_eflaw/ExpandDims
ExpandDimsfeatures_mcalpine_eflaw&mcalpine_eflaw/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
mcalpine_eflaw/ShapeShape"mcalpine_eflaw/ExpandDims:output:0*
T0*
_output_shapes
:l
"mcalpine_eflaw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$mcalpine_eflaw/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$mcalpine_eflaw/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
mcalpine_eflaw/strided_sliceStridedSlicemcalpine_eflaw/Shape:output:0+mcalpine_eflaw/strided_slice/stack:output:0-mcalpine_eflaw/strided_slice/stack_1:output:0-mcalpine_eflaw/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
mcalpine_eflaw/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :˘
mcalpine_eflaw/Reshape/shapePack%mcalpine_eflaw/strided_slice:output:0'mcalpine_eflaw/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
mcalpine_eflaw/ReshapeReshape"mcalpine_eflaw/ExpandDims:output:0%mcalpine_eflaw/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
noun_to_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
noun_to_adj/ExpandDims
ExpandDimsfeatures_noun_to_adj#noun_to_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
noun_to_adj/ShapeShapenoun_to_adj/ExpandDims:output:0*
T0*
_output_shapes
:i
noun_to_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!noun_to_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!noun_to_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
noun_to_adj/strided_sliceStridedSlicenoun_to_adj/Shape:output:0(noun_to_adj/strided_slice/stack:output:0*noun_to_adj/strided_slice/stack_1:output:0*noun_to_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
noun_to_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
noun_to_adj/Reshape/shapePack"noun_to_adj/strided_slice:output:0$noun_to_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
noun_to_adj/ReshapeReshapenoun_to_adj/ExpandDims:output:0"noun_to_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙o
$num_of_grammar_errors/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ż
 num_of_grammar_errors/ExpandDims
ExpandDimsfeatures_num_of_grammar_errors-num_of_grammar_errors/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
num_of_grammar_errors/CastCast)num_of_grammar_errors/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
num_of_grammar_errors/ShapeShapenum_of_grammar_errors/Cast:y:0*
T0*
_output_shapes
:s
)num_of_grammar_errors/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+num_of_grammar_errors/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+num_of_grammar_errors/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#num_of_grammar_errors/strided_sliceStridedSlice$num_of_grammar_errors/Shape:output:02num_of_grammar_errors/strided_slice/stack:output:04num_of_grammar_errors/strided_slice/stack_1:output:04num_of_grammar_errors/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%num_of_grammar_errors/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ˇ
#num_of_grammar_errors/Reshape/shapePack,num_of_grammar_errors/strided_slice:output:0.num_of_grammar_errors/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:¨
num_of_grammar_errors/ReshapeReshapenum_of_grammar_errors/Cast:y:0,num_of_grammar_errors/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!num_of_short_forms/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ś
num_of_short_forms/ExpandDims
ExpandDimsfeatures_num_of_short_forms*num_of_short_forms/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
num_of_short_forms/CastCast&num_of_short_forms/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙c
num_of_short_forms/ShapeShapenum_of_short_forms/Cast:y:0*
T0*
_output_shapes
:p
&num_of_short_forms/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(num_of_short_forms/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(num_of_short_forms/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 num_of_short_forms/strided_sliceStridedSlice!num_of_short_forms/Shape:output:0/num_of_short_forms/strided_slice/stack:output:01num_of_short_forms/strided_slice/stack_1:output:01num_of_short_forms/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"num_of_short_forms/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 num_of_short_forms/Reshape/shapePack)num_of_short_forms/strided_slice:output:0+num_of_short_forms/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
num_of_short_forms/ReshapeReshapenum_of_short_forms/Cast:y:0)num_of_short_forms/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#number_of_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
number_of_diff_words/ExpandDims
ExpandDimsfeatures_number_of_diff_words,number_of_diff_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
number_of_diff_words/CastCast(number_of_diff_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
number_of_diff_words/ShapeShapenumber_of_diff_words/Cast:y:0*
T0*
_output_shapes
:r
(number_of_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*number_of_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*number_of_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"number_of_diff_words/strided_sliceStridedSlice#number_of_diff_words/Shape:output:01number_of_diff_words/strided_slice/stack:output:03number_of_diff_words/strided_slice/stack_1:output:03number_of_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$number_of_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"number_of_diff_words/Reshape/shapePack+number_of_diff_words/strided_slice:output:0-number_of_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ľ
number_of_diff_words/ReshapeReshapenumber_of_diff_words/Cast:y:0+number_of_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
number_of_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
number_of_words/ExpandDims
ExpandDimsfeatures_number_of_words'number_of_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
number_of_words/CastCast#number_of_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙]
number_of_words/ShapeShapenumber_of_words/Cast:y:0*
T0*
_output_shapes
:m
#number_of_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%number_of_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%number_of_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
number_of_words/strided_sliceStridedSlicenumber_of_words/Shape:output:0,number_of_words/strided_slice/stack:output:0.number_of_words/strided_slice/stack_1:output:0.number_of_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
number_of_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
number_of_words/Reshape/shapePack&number_of_words/strided_slice:output:0(number_of_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
number_of_words/ReshapeReshapenumber_of_words/Cast:y:0&number_of_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙j
phrase_diversity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ 
phrase_diversity/ExpandDims
ExpandDimsfeatures_phrase_diversity(phrase_diversity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙j
phrase_diversity/ShapeShape$phrase_diversity/ExpandDims:output:0*
T0*
_output_shapes
:n
$phrase_diversity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&phrase_diversity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&phrase_diversity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ś
phrase_diversity/strided_sliceStridedSlicephrase_diversity/Shape:output:0-phrase_diversity/strided_slice/stack:output:0/phrase_diversity/strided_slice/stack_1:output:0/phrase_diversity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 phrase_diversity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :¨
phrase_diversity/Reshape/shapePack'phrase_diversity/strided_slice:output:0)phrase_diversity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:¤
phrase_diversity/ReshapeReshape$phrase_diversity/ExpandDims:output:0'phrase_diversity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
punctuations/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
punctuations/ExpandDims
ExpandDimsfeatures_punctuations$punctuations/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
punctuations/ShapeShape punctuations/ExpandDims:output:0*
T0*
_output_shapes
:j
 punctuations/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"punctuations/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"punctuations/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
punctuations/strided_sliceStridedSlicepunctuations/Shape:output:0)punctuations/strided_slice/stack:output:0+punctuations/strided_slice/stack_1:output:0+punctuations/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
punctuations/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
punctuations/Reshape/shapePack#punctuations/strided_slice:output:0%punctuations/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
punctuations/ReshapeReshape punctuations/ExpandDims:output:0#punctuations/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"sentence_complexity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Š
sentence_complexity/ExpandDims
ExpandDimsfeatures_sentence_complexity+sentence_complexity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
sentence_complexity/ShapeShape'sentence_complexity/ExpandDims:output:0*
T0*
_output_shapes
:q
'sentence_complexity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sentence_complexity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sentence_complexity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!sentence_complexity/strided_sliceStridedSlice"sentence_complexity/Shape:output:00sentence_complexity/strided_slice/stack:output:02sentence_complexity/strided_slice/stack_1:output:02sentence_complexity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sentence_complexity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!sentence_complexity/Reshape/shapePack*sentence_complexity/strided_slice:output:0,sentence_complexity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
sentence_complexity/ReshapeReshape'sentence_complexity/ExpandDims:output:0*sentence_complexity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!sentiment_compound/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ś
sentiment_compound/ExpandDims
ExpandDimsfeatures_sentiment_compound*sentiment_compound/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
sentiment_compound/ShapeShape&sentiment_compound/ExpandDims:output:0*
T0*
_output_shapes
:p
&sentiment_compound/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sentiment_compound/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sentiment_compound/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sentiment_compound/strided_sliceStridedSlice!sentiment_compound/Shape:output:0/sentiment_compound/strided_slice/stack:output:01sentiment_compound/strided_slice/stack_1:output:01sentiment_compound/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sentiment_compound/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 sentiment_compound/Reshape/shapePack)sentiment_compound/strided_slice:output:0+sentiment_compound/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
sentiment_compound/ReshapeReshape&sentiment_compound/ExpandDims:output:0)sentiment_compound/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!sentiment_negative/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ś
sentiment_negative/ExpandDims
ExpandDimsfeatures_sentiment_negative*sentiment_negative/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
sentiment_negative/ShapeShape&sentiment_negative/ExpandDims:output:0*
T0*
_output_shapes
:p
&sentiment_negative/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sentiment_negative/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sentiment_negative/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sentiment_negative/strided_sliceStridedSlice!sentiment_negative/Shape:output:0/sentiment_negative/strided_slice/stack:output:01sentiment_negative/strided_slice/stack_1:output:01sentiment_negative/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sentiment_negative/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 sentiment_negative/Reshape/shapePack)sentiment_negative/strided_slice:output:0+sentiment_negative/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
sentiment_negative/ReshapeReshape&sentiment_negative/ExpandDims:output:0)sentiment_negative/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!sentiment_positive/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ś
sentiment_positive/ExpandDims
ExpandDimsfeatures_sentiment_positive*sentiment_positive/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
sentiment_positive/ShapeShape&sentiment_positive/ExpandDims:output:0*
T0*
_output_shapes
:p
&sentiment_positive/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sentiment_positive/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sentiment_positive/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sentiment_positive/strided_sliceStridedSlice!sentiment_positive/Shape:output:0/sentiment_positive/strided_slice/stack:output:01sentiment_positive/strided_slice/stack_1:output:01sentiment_positive/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sentiment_positive/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 sentiment_positive/Reshape/shapePack)sentiment_positive/strided_slice:output:0+sentiment_positive/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
sentiment_positive/ReshapeReshape&sentiment_positive/ExpandDims:output:0)sentiment_positive/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"stopwords_frequency/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Š
stopwords_frequency/ExpandDims
ExpandDimsfeatures_stopwords_frequency+stopwords_frequency/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
stopwords_frequency/ShapeShape'stopwords_frequency/ExpandDims:output:0*
T0*
_output_shapes
:q
'stopwords_frequency/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)stopwords_frequency/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)stopwords_frequency/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!stopwords_frequency/strided_sliceStridedSlice"stopwords_frequency/Shape:output:00stopwords_frequency/strided_slice/stack:output:02stopwords_frequency/strided_slice/stack_1:output:02stopwords_frequency/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#stopwords_frequency/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!stopwords_frequency/Reshape/shapePack*stopwords_frequency/strided_slice:output:0,stopwords_frequency/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
stopwords_frequency/ReshapeReshape'stopwords_frequency/ExpandDims:output:0*stopwords_frequency/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
text_standard/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
text_standard/ExpandDims
ExpandDimsfeatures_text_standard%text_standard/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d
text_standard/ShapeShape!text_standard/ExpandDims:output:0*
T0*
_output_shapes
:k
!text_standard/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#text_standard/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#text_standard/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
text_standard/strided_sliceStridedSlicetext_standard/Shape:output:0*text_standard/strided_slice/stack:output:0,text_standard/strided_slice/stack_1:output:0,text_standard/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
text_standard/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
text_standard/Reshape/shapePack$text_standard/strided_slice:output:0&text_standard/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
text_standard/ReshapeReshape!text_standard/ExpandDims:output:0$text_standard/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙]
ttr/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙y
ttr/ExpandDims
ExpandDimsfeatures_ttrttr/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P
	ttr/ShapeShapettr/ExpandDims:output:0*
T0*
_output_shapes
:a
ttr/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
ttr/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
ttr/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
ttr/strided_sliceStridedSlicettr/Shape:output:0 ttr/strided_slice/stack:output:0"ttr/strided_slice/stack_1:output:0"ttr/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
ttr/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ttr/Reshape/shapePackttr/strided_slice:output:0ttr/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:}
ttr/ReshapeReshapettr/ExpandDims:output:0ttr/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
verb_to_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
verb_to_adv/ExpandDims
ExpandDimsfeatures_verb_to_adv#verb_to_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
verb_to_adv/ShapeShapeverb_to_adv/ExpandDims:output:0*
T0*
_output_shapes
:i
verb_to_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!verb_to_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!verb_to_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
verb_to_adv/strided_sliceStridedSliceverb_to_adv/Shape:output:0(verb_to_adv/strided_slice/stack:output:0*verb_to_adv/strided_slice/stack_1:output:0*verb_to_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
verb_to_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
verb_to_adv/Reshape/shapePack"verb_to_adv/strided_slice:output:0$verb_to_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
verb_to_adv/ReshapeReshapeverb_to_adv/ExpandDims:output:0"verb_to_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź

concatConcatV2ARI/Reshape:output:0%Incorrect_form_ratio/Reshape:output:0 av_word_per_sen/Reshape:output:0 coherence_score/Reshape:output:0-dale_chall_readability_score/Reshape:output:0%flesch_kincaid_grade/Reshape:output:0$flesch_reading_ease/Reshape:output:0 freq_diff_words/Reshape:output:0freq_of_adj/Reshape:output:0freq_of_adv/Reshape:output:0%freq_of_distinct_adj/Reshape:output:0%freq_of_distinct_adv/Reshape:output:0freq_of_noun/Reshape:output:0 freq_of_pronoun/Reshape:output:0#freq_of_transition/Reshape:output:0freq_of_verb/Reshape:output:0$freq_of_wrong_words/Reshape:output:0%lexrank_avg_min_diff/Reshape:output:0&lexrank_interquartile/Reshape:output:0mcalpine_eflaw/Reshape:output:0noun_to_adj/Reshape:output:0&num_of_grammar_errors/Reshape:output:0#num_of_short_forms/Reshape:output:0%number_of_diff_words/Reshape:output:0 number_of_words/Reshape:output:0!phrase_diversity/Reshape:output:0punctuations/Reshape:output:0$sentence_complexity/Reshape:output:0#sentiment_compound/Reshape:output:0#sentiment_negative/Reshape:output:0#sentiment_positive/Reshape:output:0$stopwords_frequency/Reshape:output:0text_standard/Reshape:output:0ttr/Reshape:output:0verb_to_adv/Reshape:output:0concat/axis:output:0*
N#*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙#W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙#"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ŕ
_input_shapesŽ
Ť:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:Q M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefeatures/ARI:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/Incorrect_form_ratio:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/av_word_per_sen:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/coherence_score:VR
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_namefeatures/cohesion:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_namefeatures/corrected_text:jf
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
?
_user_specified_name'%features/dale_chall_readability_score:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/flesch_kincaid_grade:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/flesch_reading_ease:]	Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/freq_diff_words:Y
U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/freq_of_adj:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/freq_of_adv:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/freq_of_distinct_adj:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/freq_of_distinct_adv:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namefeatures/freq_of_noun:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/freq_of_pronoun:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/freq_of_transition:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namefeatures/freq_of_verb:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/freq_of_wrong_words:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/lexrank_avg_min_diff:c_
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name features/lexrank_interquartile:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_namefeatures/mcalpine_eflaw:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/noun_to_adj:c_
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name features/num_of_grammar_errors:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/num_of_short_forms:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/number_of_diff_words:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/number_of_words:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_namefeatures/phrase_diversity:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namefeatures/punctuations:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/sentence_complexity:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/sentiment_compound:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/sentiment_negative:` \
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/sentiment_positive:a!]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/stopwords_frequency:["W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_namefeatures/text_standard:Q#M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefeatures/ttr:Y$U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/verb_to_adv
×0

#__inference_signature_wrapper_52452
ari	
incorrect_form_ratio
av_word_per_sen
coherence_score
cohesion
corrected_text 
dale_chall_readability_score
flesch_kincaid_grade
flesch_reading_ease
freq_diff_words
freq_of_adj
freq_of_adv
freq_of_distinct_adj
freq_of_distinct_adv
freq_of_noun
freq_of_pronoun
freq_of_transition
freq_of_verb
freq_of_wrong_words
lexrank_avg_min_diff
lexrank_interquartile
mcalpine_eflaw
noun_to_adj
num_of_grammar_errors	
num_of_short_forms	
number_of_diff_words	
number_of_words	
phrase_diversity
punctuations
sentence_complexity
sentiment_compound
sentiment_negative
sentiment_positive
stopwords_frequency
text_standard
ttr
verb_to_adv
unknown:#
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallariincorrect_form_ratioav_word_per_sencoherence_scorecohesioncorrected_textdale_chall_readability_scoreflesch_kincaid_gradeflesch_reading_easefreq_diff_wordsfreq_of_adjfreq_of_advfreq_of_distinct_adjfreq_of_distinct_advfreq_of_nounfreq_of_pronounfreq_of_transitionfreq_of_verbfreq_of_wrong_wordslexrank_avg_min_difflexrank_interquartilemcalpine_eflawnoun_to_adjnum_of_grammar_errorsnum_of_short_formsnumber_of_diff_wordsnumber_of_wordsphrase_diversitypunctuationssentence_complexitysentiment_compoundsentiment_negativesentiment_positivestopwords_frequencytext_standardttrverb_to_advunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*6
Tin/
-2+					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

%&'()**-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_51010o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ě
_input_shapesş
ˇ:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameARI:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameIncorrect_form_ratio:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameav_word_per_sen:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namecoherence_score:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
cohesion:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namecorrected_text:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namedale_chall_readability_score:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameflesch_kincaid_grade:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameflesch_reading_ease:T	P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefreq_diff_words:P
L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefreq_of_adj:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefreq_of_adv:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefreq_of_distinct_adj:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefreq_of_distinct_adv:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefreq_of_noun:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefreq_of_pronoun:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namefreq_of_transition:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefreq_of_verb:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namefreq_of_wrong_words:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namelexrank_avg_min_diff:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namelexrank_interquartile:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namemcalpine_eflaw:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namenoun_to_adj:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namenum_of_grammar_errors:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namenum_of_short_forms:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namenumber_of_diff_words:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namenumber_of_words:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namephrase_diversity:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepunctuations:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namesentence_complexity:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_compound:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_negative:W S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_positive:X!T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namestopwords_frequency:R"N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nametext_standard:H#D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namettr:P$L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameverb_to_adv
1

,__inference_sequential_2_layer_call_fn_52279
ari	
incorrect_form_ratio
av_word_per_sen
coherence_score
cohesion
corrected_text 
dale_chall_readability_score
flesch_kincaid_grade
flesch_reading_ease
freq_diff_words
freq_of_adj
freq_of_adv
freq_of_distinct_adj
freq_of_distinct_adv
freq_of_noun
freq_of_pronoun
freq_of_transition
freq_of_verb
freq_of_wrong_words
lexrank_avg_min_diff
lexrank_interquartile
mcalpine_eflaw
noun_to_adj
num_of_grammar_errors	
num_of_short_forms	
number_of_diff_words	
number_of_words	
phrase_diversity
punctuations
sentence_complexity
sentiment_compound
sentiment_negative
sentiment_positive
stopwords_frequency
text_standard
ttr
verb_to_adv
unknown:#
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity˘StatefulPartitionedCallź
StatefulPartitionedCallStatefulPartitionedCallariincorrect_form_ratioav_word_per_sencoherence_scorecohesioncorrected_textdale_chall_readability_scoreflesch_kincaid_gradeflesch_reading_easefreq_diff_wordsfreq_of_adjfreq_of_advfreq_of_distinct_adjfreq_of_distinct_advfreq_of_nounfreq_of_pronounfreq_of_transitionfreq_of_verbfreq_of_wrong_wordslexrank_avg_min_difflexrank_interquartilemcalpine_eflawnoun_to_adjnum_of_grammar_errorsnum_of_short_formsnumber_of_diff_wordsnumber_of_wordsphrase_diversitypunctuationssentence_complexitysentiment_compoundsentiment_negativesentiment_positivestopwords_frequencytext_standardttrverb_to_advunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*6
Tin/
-2+					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

%&'()**-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_52211o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ě
_input_shapesş
ˇ:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:H D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameARI:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameIncorrect_form_ratio:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameav_word_per_sen:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namecoherence_score:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
cohesion:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namecorrected_text:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namedale_chall_readability_score:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameflesch_kincaid_grade:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameflesch_reading_ease:T	P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefreq_diff_words:P
L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefreq_of_adj:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefreq_of_adv:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefreq_of_distinct_adj:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefreq_of_distinct_adv:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefreq_of_noun:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefreq_of_pronoun:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namefreq_of_transition:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefreq_of_verb:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namefreq_of_wrong_words:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namelexrank_avg_min_diff:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namelexrank_interquartile:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namemcalpine_eflaw:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namenoun_to_adj:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namenum_of_grammar_errors:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namenum_of_short_forms:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namenumber_of_diff_words:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namenumber_of_words:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namephrase_diversity:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepunctuations:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namesentence_complexity:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_compound:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_negative:W S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_positive:X!T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namestopwords_frequency:R"N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nametext_standard:H#D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namettr:P$L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameverb_to_adv
Ć<
	
G__inference_sequential_2_layer_call_and_return_conditional_losses_52391
ari	
incorrect_form_ratio
av_word_per_sen
coherence_score
cohesion
corrected_text 
dale_chall_readability_score
flesch_kincaid_grade
flesch_reading_ease
freq_diff_words
freq_of_adj
freq_of_adv
freq_of_distinct_adj
freq_of_distinct_adv
freq_of_noun
freq_of_pronoun
freq_of_transition
freq_of_verb
freq_of_wrong_words
lexrank_avg_min_diff
lexrank_interquartile
mcalpine_eflaw
noun_to_adj
num_of_grammar_errors	
num_of_short_forms	
number_of_diff_words	
number_of_words	
phrase_diversity
punctuations
sentence_complexity
sentiment_compound
sentiment_negative
sentiment_positive
stopwords_frequency
text_standard
ttr
verb_to_adv
hidden0_52375:#
hidden0_52377:
hidden1_52380:
hidden1_52382:
output_52385:
output_52387:
identity˘Hidden0/StatefulPartitionedCall˘Hidden1/StatefulPartitionedCall˘Output/StatefulPartitionedCallď
dense_features/PartitionedCallPartitionedCallariincorrect_form_ratioav_word_per_sencoherence_scorecohesioncorrected_textdale_chall_readability_scoreflesch_kincaid_gradeflesch_reading_easefreq_diff_wordsfreq_of_adjfreq_of_advfreq_of_distinct_adjfreq_of_distinct_advfreq_of_nounfreq_of_pronounfreq_of_transitionfreq_of_verbfreq_of_wrong_wordslexrank_avg_min_difflexrank_interquartilemcalpine_eflawnoun_to_adjnum_of_grammar_errorsnum_of_short_formsnumber_of_diff_wordsnumber_of_wordsphrase_diversitypunctuationssentence_complexitysentiment_compoundsentiment_negativesentiment_positivestopwords_frequencytext_standardttrverb_to_adv*0
Tin)
'2%					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙#* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_52061
Hidden0/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0hidden0_52375hidden0_52377*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Hidden0_layer_call_and_return_conditional_losses_51499
Hidden1/StatefulPartitionedCallStatefulPartitionedCall(Hidden0/StatefulPartitionedCall:output:0hidden1_52380hidden1_52382*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Hidden1_layer_call_and_return_conditional_losses_51516
Output/StatefulPartitionedCallStatefulPartitionedCall(Hidden1/StatefulPartitionedCall:output:0output_52385output_52387*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_51532v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
NoOpNoOp ^Hidden0/StatefulPartitionedCall ^Hidden1/StatefulPartitionedCall^Output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ě
_input_shapesş
ˇ:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : 2B
Hidden0/StatefulPartitionedCallHidden0/StatefulPartitionedCall2B
Hidden1/StatefulPartitionedCallHidden1/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:H D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameARI:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameIncorrect_form_ratio:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameav_word_per_sen:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namecoherence_score:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
cohesion:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namecorrected_text:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namedale_chall_readability_score:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameflesch_kincaid_grade:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameflesch_reading_ease:T	P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefreq_diff_words:P
L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefreq_of_adj:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefreq_of_adv:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefreq_of_distinct_adj:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefreq_of_distinct_adv:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefreq_of_noun:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefreq_of_pronoun:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namefreq_of_transition:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefreq_of_verb:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namefreq_of_wrong_words:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namelexrank_avg_min_diff:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namelexrank_interquartile:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namemcalpine_eflaw:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namenoun_to_adj:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namenum_of_grammar_errors:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namenum_of_short_forms:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namenumber_of_diff_words:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namenumber_of_words:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namephrase_diversity:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepunctuations:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namesentence_complexity:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_compound:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_negative:W S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_positive:X!T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namestopwords_frequency:R"N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nametext_standard:H#D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namettr:P$L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameverb_to_adv
Ä	
ň
A__inference_Output_layer_call_and_return_conditional_losses_51532

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
7
 

,__inference_sequential_2_layer_call_fn_52558

inputs_ari	
inputs_incorrect_form_ratio
inputs_av_word_per_sen
inputs_coherence_score
inputs_cohesion
inputs_corrected_text'
#inputs_dale_chall_readability_score
inputs_flesch_kincaid_grade
inputs_flesch_reading_ease
inputs_freq_diff_words
inputs_freq_of_adj
inputs_freq_of_adv
inputs_freq_of_distinct_adj
inputs_freq_of_distinct_adv
inputs_freq_of_noun
inputs_freq_of_pronoun
inputs_freq_of_transition
inputs_freq_of_verb
inputs_freq_of_wrong_words
inputs_lexrank_avg_min_diff 
inputs_lexrank_interquartile
inputs_mcalpine_eflaw
inputs_noun_to_adj 
inputs_num_of_grammar_errors	
inputs_num_of_short_forms	
inputs_number_of_diff_words	
inputs_number_of_words	
inputs_phrase_diversity
inputs_punctuations
inputs_sentence_complexity
inputs_sentiment_compound
inputs_sentiment_negative
inputs_sentiment_positive
inputs_stopwords_frequency
inputs_text_standard

inputs_ttr
inputs_verb_to_adv
unknown:#
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity˘StatefulPartitionedCallż

StatefulPartitionedCallStatefulPartitionedCall
inputs_ariinputs_incorrect_form_ratioinputs_av_word_per_seninputs_coherence_scoreinputs_cohesioninputs_corrected_text#inputs_dale_chall_readability_scoreinputs_flesch_kincaid_gradeinputs_flesch_reading_easeinputs_freq_diff_wordsinputs_freq_of_adjinputs_freq_of_advinputs_freq_of_distinct_adjinputs_freq_of_distinct_advinputs_freq_of_nouninputs_freq_of_pronouninputs_freq_of_transitioninputs_freq_of_verbinputs_freq_of_wrong_wordsinputs_lexrank_avg_min_diffinputs_lexrank_interquartileinputs_mcalpine_eflawinputs_noun_to_adjinputs_num_of_grammar_errorsinputs_num_of_short_formsinputs_number_of_diff_wordsinputs_number_of_wordsinputs_phrase_diversityinputs_punctuationsinputs_sentence_complexityinputs_sentiment_compoundinputs_sentiment_negativeinputs_sentiment_positiveinputs_stopwords_frequencyinputs_text_standard
inputs_ttrinputs_verb_to_advunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*6
Tin/
-2+					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*(
_read_only_resource_inputs

%&'()**-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_sequential_2_layer_call_and_return_conditional_losses_52211o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ě
_input_shapesş
ˇ:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs/ARI:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/Incorrect_form_ratio:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/av_word_per_sen:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/coherence_score:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs/cohesion:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs/corrected_text:hd
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
=
_user_specified_name%#inputs/dale_chall_readability_score:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/flesch_kincaid_grade:_[
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/flesch_reading_ease:[	W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/freq_diff_words:W
S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/freq_of_adj:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/freq_of_adv:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/freq_of_distinct_adj:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/freq_of_distinct_adv:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs/freq_of_noun:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/freq_of_pronoun:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/freq_of_transition:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs/freq_of_verb:_[
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/freq_of_wrong_words:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/lexrank_avg_min_diff:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_nameinputs/lexrank_interquartile:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs/mcalpine_eflaw:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/noun_to_adj:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_nameinputs/num_of_grammar_errors:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/num_of_short_forms:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/number_of_diff_words:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/number_of_words:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs/phrase_diversity:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs/punctuations:_[
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/sentence_complexity:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/sentiment_compound:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/sentiment_negative:^ Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/sentiment_positive:_![
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/stopwords_frequency:Y"U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs/text_standard:O#K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs/ttr:W$S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/verb_to_adv


ó
B__inference_Hidden1_layer_call_and_return_conditional_losses_51516

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ô5

G__inference_sequential_2_layer_call_and_return_conditional_losses_51539

inputs	
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
	inputs_14
	inputs_15
	inputs_16
	inputs_17
	inputs_18
	inputs_19
	inputs_20
	inputs_21
	inputs_22
	inputs_23	
	inputs_24	
	inputs_25	
	inputs_26	
	inputs_27
	inputs_28
	inputs_29
	inputs_30
	inputs_31
	inputs_32
	inputs_33
	inputs_34
	inputs_35
	inputs_36
hidden0_51500:#
hidden0_51502:
hidden1_51517:
hidden1_51519:
output_51533:
output_51535:
identity˘Hidden0/StatefulPartitionedCall˘Hidden1/StatefulPartitionedCall˘Output/StatefulPartitionedCallę
dense_features/PartitionedCallPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13	inputs_14	inputs_15	inputs_16	inputs_17	inputs_18	inputs_19	inputs_20	inputs_21	inputs_22	inputs_23	inputs_24	inputs_25	inputs_26	inputs_27	inputs_28	inputs_29	inputs_30	inputs_31	inputs_32	inputs_33	inputs_34	inputs_35	inputs_36*0
Tin)
'2%					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙#* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_51486
Hidden0/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0hidden0_51500hidden0_51502*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Hidden0_layer_call_and_return_conditional_losses_51499
Hidden1/StatefulPartitionedCallStatefulPartitionedCall(Hidden0/StatefulPartitionedCall:output:0hidden1_51517hidden1_51519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Hidden1_layer_call_and_return_conditional_losses_51516
Output/StatefulPartitionedCallStatefulPartitionedCall(Hidden1/StatefulPartitionedCall:output:0output_51533output_51535*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_51532v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
NoOpNoOp ^Hidden0/StatefulPartitionedCall ^Hidden1/StatefulPartitionedCall^Output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ě
_input_shapesş
ˇ:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : 2B
Hidden0/StatefulPartitionedCallHidden0/StatefulPartitionedCall2B
Hidden1/StatefulPartitionedCallHidden1/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K	G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K
G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:KG
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K!G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K"G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K#G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs:K$G
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ś5
Ż	
.__inference_dense_features_layer_call_fn_53474
features_ari	!
features_incorrect_form_ratio
features_av_word_per_sen
features_coherence_score
features_cohesion
features_corrected_text)
%features_dale_chall_readability_score!
features_flesch_kincaid_grade 
features_flesch_reading_ease
features_freq_diff_words
features_freq_of_adj
features_freq_of_adv!
features_freq_of_distinct_adj!
features_freq_of_distinct_adv
features_freq_of_noun
features_freq_of_pronoun
features_freq_of_transition
features_freq_of_verb 
features_freq_of_wrong_words!
features_lexrank_avg_min_diff"
features_lexrank_interquartile
features_mcalpine_eflaw
features_noun_to_adj"
features_num_of_grammar_errors	
features_num_of_short_forms	!
features_number_of_diff_words	
features_number_of_words	
features_phrase_diversity
features_punctuations 
features_sentence_complexity
features_sentiment_compound
features_sentiment_negative
features_sentiment_positive 
features_stopwords_frequency
features_text_standard
features_ttr
features_verb_to_adv
identity­

PartitionedCallPartitionedCallfeatures_arifeatures_incorrect_form_ratiofeatures_av_word_per_senfeatures_coherence_scorefeatures_cohesionfeatures_corrected_text%features_dale_chall_readability_scorefeatures_flesch_kincaid_gradefeatures_flesch_reading_easefeatures_freq_diff_wordsfeatures_freq_of_adjfeatures_freq_of_advfeatures_freq_of_distinct_adjfeatures_freq_of_distinct_advfeatures_freq_of_nounfeatures_freq_of_pronounfeatures_freq_of_transitionfeatures_freq_of_verbfeatures_freq_of_wrong_wordsfeatures_lexrank_avg_min_difffeatures_lexrank_interquartilefeatures_mcalpine_eflawfeatures_noun_to_adjfeatures_num_of_grammar_errorsfeatures_num_of_short_formsfeatures_number_of_diff_wordsfeatures_number_of_wordsfeatures_phrase_diversityfeatures_punctuationsfeatures_sentence_complexityfeatures_sentiment_compoundfeatures_sentiment_negativefeatures_sentiment_positivefeatures_stopwords_frequencyfeatures_text_standardfeatures_ttrfeatures_verb_to_adv*0
Tin)
'2%					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙#* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_52061`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙#"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ŕ
_input_shapesŽ
Ť:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:Q M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefeatures/ARI:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/Incorrect_form_ratio:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/av_word_per_sen:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/coherence_score:VR
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_namefeatures/cohesion:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_namefeatures/corrected_text:jf
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
?
_user_specified_name'%features/dale_chall_readability_score:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/flesch_kincaid_grade:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/flesch_reading_ease:]	Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/freq_diff_words:Y
U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/freq_of_adj:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/freq_of_adv:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/freq_of_distinct_adj:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/freq_of_distinct_adv:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namefeatures/freq_of_noun:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/freq_of_pronoun:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/freq_of_transition:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namefeatures/freq_of_verb:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/freq_of_wrong_words:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/lexrank_avg_min_diff:c_
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name features/lexrank_interquartile:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_namefeatures/mcalpine_eflaw:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/noun_to_adj:c_
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name features/num_of_grammar_errors:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/num_of_short_forms:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/number_of_diff_words:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/number_of_words:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_namefeatures/phrase_diversity:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namefeatures/punctuations:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/sentence_complexity:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/sentiment_compound:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/sentiment_negative:` \
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/sentiment_positive:a!]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/stopwords_frequency:["W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_namefeatures/text_standard:Q#M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefeatures/ttr:Y$U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/verb_to_adv
Č
Ę	
I__inference_dense_features_layer_call_and_return_conditional_losses_53871
features_ari	!
features_incorrect_form_ratio
features_av_word_per_sen
features_coherence_score
features_cohesion
features_corrected_text)
%features_dale_chall_readability_score!
features_flesch_kincaid_grade 
features_flesch_reading_ease
features_freq_diff_words
features_freq_of_adj
features_freq_of_adv!
features_freq_of_distinct_adj!
features_freq_of_distinct_adv
features_freq_of_noun
features_freq_of_pronoun
features_freq_of_transition
features_freq_of_verb 
features_freq_of_wrong_words!
features_lexrank_avg_min_diff"
features_lexrank_interquartile
features_mcalpine_eflaw
features_noun_to_adj"
features_num_of_grammar_errors	
features_num_of_short_forms	!
features_number_of_diff_words	
features_number_of_words	
features_phrase_diversity
features_punctuations 
features_sentence_complexity
features_sentiment_compound
features_sentiment_negative
features_sentiment_positive 
features_stopwords_frequency
features_text_standard
features_ttr
features_verb_to_adv
identity]
ARI/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙y
ARI/ExpandDims
ExpandDimsfeatures_ariARI/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙j
ARI/CastCastARI/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙E
	ARI/ShapeShapeARI/Cast:y:0*
T0*
_output_shapes
:a
ARI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
ARI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
ARI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
ARI/strided_sliceStridedSliceARI/Shape:output:0 ARI/strided_slice/stack:output:0"ARI/strided_slice/stack_1:output:0"ARI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
ARI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ARI/Reshape/shapePackARI/strided_slice:output:0ARI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
ARI/ReshapeReshapeARI/Cast:y:0ARI/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#Incorrect_form_ratio/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
Incorrect_form_ratio/ExpandDims
ExpandDimsfeatures_incorrect_form_ratio,Incorrect_form_ratio/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
Incorrect_form_ratio/ShapeShape(Incorrect_form_ratio/ExpandDims:output:0*
T0*
_output_shapes
:r
(Incorrect_form_ratio/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*Incorrect_form_ratio/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*Incorrect_form_ratio/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"Incorrect_form_ratio/strided_sliceStridedSlice#Incorrect_form_ratio/Shape:output:01Incorrect_form_ratio/strided_slice/stack:output:03Incorrect_form_ratio/strided_slice/stack_1:output:03Incorrect_form_ratio/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$Incorrect_form_ratio/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"Incorrect_form_ratio/Reshape/shapePack+Incorrect_form_ratio/strided_slice:output:0-Incorrect_form_ratio/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
Incorrect_form_ratio/ReshapeReshape(Incorrect_form_ratio/ExpandDims:output:0+Incorrect_form_ratio/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
av_word_per_sen/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
av_word_per_sen/ExpandDims
ExpandDimsfeatures_av_word_per_sen'av_word_per_sen/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
av_word_per_sen/ShapeShape#av_word_per_sen/ExpandDims:output:0*
T0*
_output_shapes
:m
#av_word_per_sen/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%av_word_per_sen/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%av_word_per_sen/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
av_word_per_sen/strided_sliceStridedSliceav_word_per_sen/Shape:output:0,av_word_per_sen/strided_slice/stack:output:0.av_word_per_sen/strided_slice/stack_1:output:0.av_word_per_sen/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
av_word_per_sen/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
av_word_per_sen/Reshape/shapePack&av_word_per_sen/strided_slice:output:0(av_word_per_sen/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
av_word_per_sen/ReshapeReshape#av_word_per_sen/ExpandDims:output:0&av_word_per_sen/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
coherence_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
coherence_score/ExpandDims
ExpandDimsfeatures_coherence_score'coherence_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
coherence_score/ShapeShape#coherence_score/ExpandDims:output:0*
T0*
_output_shapes
:m
#coherence_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%coherence_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%coherence_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
coherence_score/strided_sliceStridedSlicecoherence_score/Shape:output:0,coherence_score/strided_slice/stack:output:0.coherence_score/strided_slice/stack_1:output:0.coherence_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
coherence_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
coherence_score/Reshape/shapePack&coherence_score/strided_slice:output:0(coherence_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
coherence_score/ReshapeReshape#coherence_score/ExpandDims:output:0&coherence_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙v
+dale_chall_readability_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ä
'dale_chall_readability_score/ExpandDims
ExpandDims%features_dale_chall_readability_score4dale_chall_readability_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"dale_chall_readability_score/ShapeShape0dale_chall_readability_score/ExpandDims:output:0*
T0*
_output_shapes
:z
0dale_chall_readability_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dale_chall_readability_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dale_chall_readability_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dale_chall_readability_score/strided_sliceStridedSlice+dale_chall_readability_score/Shape:output:09dale_chall_readability_score/strided_slice/stack:output:0;dale_chall_readability_score/strided_slice/stack_1:output:0;dale_chall_readability_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,dale_chall_readability_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ě
*dale_chall_readability_score/Reshape/shapePack3dale_chall_readability_score/strided_slice:output:05dale_chall_readability_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Č
$dale_chall_readability_score/ReshapeReshape0dale_chall_readability_score/ExpandDims:output:03dale_chall_readability_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#flesch_kincaid_grade/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
flesch_kincaid_grade/ExpandDims
ExpandDimsfeatures_flesch_kincaid_grade,flesch_kincaid_grade/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
flesch_kincaid_grade/ShapeShape(flesch_kincaid_grade/ExpandDims:output:0*
T0*
_output_shapes
:r
(flesch_kincaid_grade/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*flesch_kincaid_grade/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*flesch_kincaid_grade/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"flesch_kincaid_grade/strided_sliceStridedSlice#flesch_kincaid_grade/Shape:output:01flesch_kincaid_grade/strided_slice/stack:output:03flesch_kincaid_grade/strided_slice/stack_1:output:03flesch_kincaid_grade/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$flesch_kincaid_grade/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"flesch_kincaid_grade/Reshape/shapePack+flesch_kincaid_grade/strided_slice:output:0-flesch_kincaid_grade/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
flesch_kincaid_grade/ReshapeReshape(flesch_kincaid_grade/ExpandDims:output:0+flesch_kincaid_grade/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"flesch_reading_ease/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Š
flesch_reading_ease/ExpandDims
ExpandDimsfeatures_flesch_reading_ease+flesch_reading_ease/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
flesch_reading_ease/ShapeShape'flesch_reading_ease/ExpandDims:output:0*
T0*
_output_shapes
:q
'flesch_reading_ease/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)flesch_reading_ease/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)flesch_reading_ease/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!flesch_reading_ease/strided_sliceStridedSlice"flesch_reading_ease/Shape:output:00flesch_reading_ease/strided_slice/stack:output:02flesch_reading_ease/strided_slice/stack_1:output:02flesch_reading_ease/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#flesch_reading_ease/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!flesch_reading_ease/Reshape/shapePack*flesch_reading_ease/strided_slice:output:0,flesch_reading_ease/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
flesch_reading_ease/ReshapeReshape'flesch_reading_ease/ExpandDims:output:0*flesch_reading_ease/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
freq_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_diff_words/ExpandDims
ExpandDimsfeatures_freq_diff_words'freq_diff_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
freq_diff_words/ShapeShape#freq_diff_words/ExpandDims:output:0*
T0*
_output_shapes
:m
#freq_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%freq_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%freq_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
freq_diff_words/strided_sliceStridedSlicefreq_diff_words/Shape:output:0,freq_diff_words/strided_slice/stack:output:0.freq_diff_words/strided_slice/stack_1:output:0.freq_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
freq_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
freq_diff_words/Reshape/shapePack&freq_diff_words/strided_slice:output:0(freq_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
freq_diff_words/ReshapeReshape#freq_diff_words/ExpandDims:output:0&freq_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
freq_of_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_adj/ExpandDims
ExpandDimsfeatures_freq_of_adj#freq_of_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
freq_of_adj/ShapeShapefreq_of_adj/ExpandDims:output:0*
T0*
_output_shapes
:i
freq_of_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!freq_of_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!freq_of_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_adj/strided_sliceStridedSlicefreq_of_adj/Shape:output:0(freq_of_adj/strided_slice/stack:output:0*freq_of_adj/strided_slice/stack_1:output:0*freq_of_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
freq_of_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_adj/Reshape/shapePack"freq_of_adj/strided_slice:output:0$freq_of_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_adj/ReshapeReshapefreq_of_adj/ExpandDims:output:0"freq_of_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
freq_of_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_adv/ExpandDims
ExpandDimsfeatures_freq_of_adv#freq_of_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
freq_of_adv/ShapeShapefreq_of_adv/ExpandDims:output:0*
T0*
_output_shapes
:i
freq_of_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!freq_of_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!freq_of_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_adv/strided_sliceStridedSlicefreq_of_adv/Shape:output:0(freq_of_adv/strided_slice/stack:output:0*freq_of_adv/strided_slice/stack_1:output:0*freq_of_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
freq_of_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_adv/Reshape/shapePack"freq_of_adv/strided_slice:output:0$freq_of_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_adv/ReshapeReshapefreq_of_adv/ExpandDims:output:0"freq_of_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#freq_of_distinct_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
freq_of_distinct_adj/ExpandDims
ExpandDimsfeatures_freq_of_distinct_adj,freq_of_distinct_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
freq_of_distinct_adj/ShapeShape(freq_of_distinct_adj/ExpandDims:output:0*
T0*
_output_shapes
:r
(freq_of_distinct_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*freq_of_distinct_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*freq_of_distinct_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"freq_of_distinct_adj/strided_sliceStridedSlice#freq_of_distinct_adj/Shape:output:01freq_of_distinct_adj/strided_slice/stack:output:03freq_of_distinct_adj/strided_slice/stack_1:output:03freq_of_distinct_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$freq_of_distinct_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"freq_of_distinct_adj/Reshape/shapePack+freq_of_distinct_adj/strided_slice:output:0-freq_of_distinct_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
freq_of_distinct_adj/ReshapeReshape(freq_of_distinct_adj/ExpandDims:output:0+freq_of_distinct_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#freq_of_distinct_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
freq_of_distinct_adv/ExpandDims
ExpandDimsfeatures_freq_of_distinct_adv,freq_of_distinct_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
freq_of_distinct_adv/ShapeShape(freq_of_distinct_adv/ExpandDims:output:0*
T0*
_output_shapes
:r
(freq_of_distinct_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*freq_of_distinct_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*freq_of_distinct_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"freq_of_distinct_adv/strided_sliceStridedSlice#freq_of_distinct_adv/Shape:output:01freq_of_distinct_adv/strided_slice/stack:output:03freq_of_distinct_adv/strided_slice/stack_1:output:03freq_of_distinct_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$freq_of_distinct_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"freq_of_distinct_adv/Reshape/shapePack+freq_of_distinct_adv/strided_slice:output:0-freq_of_distinct_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
freq_of_distinct_adv/ReshapeReshape(freq_of_distinct_adv/ExpandDims:output:0+freq_of_distinct_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
freq_of_noun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_noun/ExpandDims
ExpandDimsfeatures_freq_of_noun$freq_of_noun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
freq_of_noun/ShapeShape freq_of_noun/ExpandDims:output:0*
T0*
_output_shapes
:j
 freq_of_noun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"freq_of_noun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"freq_of_noun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_noun/strided_sliceStridedSlicefreq_of_noun/Shape:output:0)freq_of_noun/strided_slice/stack:output:0+freq_of_noun/strided_slice/stack_1:output:0+freq_of_noun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
freq_of_noun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_noun/Reshape/shapePack#freq_of_noun/strided_slice:output:0%freq_of_noun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_noun/ReshapeReshape freq_of_noun/ExpandDims:output:0#freq_of_noun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
freq_of_pronoun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_pronoun/ExpandDims
ExpandDimsfeatures_freq_of_pronoun'freq_of_pronoun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
freq_of_pronoun/ShapeShape#freq_of_pronoun/ExpandDims:output:0*
T0*
_output_shapes
:m
#freq_of_pronoun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%freq_of_pronoun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%freq_of_pronoun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
freq_of_pronoun/strided_sliceStridedSlicefreq_of_pronoun/Shape:output:0,freq_of_pronoun/strided_slice/stack:output:0.freq_of_pronoun/strided_slice/stack_1:output:0.freq_of_pronoun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
freq_of_pronoun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
freq_of_pronoun/Reshape/shapePack&freq_of_pronoun/strided_slice:output:0(freq_of_pronoun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ą
freq_of_pronoun/ReshapeReshape#freq_of_pronoun/ExpandDims:output:0&freq_of_pronoun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!freq_of_transition/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ś
freq_of_transition/ExpandDims
ExpandDimsfeatures_freq_of_transition*freq_of_transition/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
freq_of_transition/ShapeShape&freq_of_transition/ExpandDims:output:0*
T0*
_output_shapes
:p
&freq_of_transition/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(freq_of_transition/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(freq_of_transition/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 freq_of_transition/strided_sliceStridedSlice!freq_of_transition/Shape:output:0/freq_of_transition/strided_slice/stack:output:01freq_of_transition/strided_slice/stack_1:output:01freq_of_transition/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"freq_of_transition/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 freq_of_transition/Reshape/shapePack)freq_of_transition/strided_slice:output:0+freq_of_transition/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
freq_of_transition/ReshapeReshape&freq_of_transition/ExpandDims:output:0)freq_of_transition/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
freq_of_verb/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
freq_of_verb/ExpandDims
ExpandDimsfeatures_freq_of_verb$freq_of_verb/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
freq_of_verb/ShapeShape freq_of_verb/ExpandDims:output:0*
T0*
_output_shapes
:j
 freq_of_verb/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"freq_of_verb/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"freq_of_verb/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
freq_of_verb/strided_sliceStridedSlicefreq_of_verb/Shape:output:0)freq_of_verb/strided_slice/stack:output:0+freq_of_verb/strided_slice/stack_1:output:0+freq_of_verb/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
freq_of_verb/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
freq_of_verb/Reshape/shapePack#freq_of_verb/strided_slice:output:0%freq_of_verb/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
freq_of_verb/ReshapeReshape freq_of_verb/ExpandDims:output:0#freq_of_verb/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"freq_of_wrong_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Š
freq_of_wrong_words/ExpandDims
ExpandDimsfeatures_freq_of_wrong_words+freq_of_wrong_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
freq_of_wrong_words/ShapeShape'freq_of_wrong_words/ExpandDims:output:0*
T0*
_output_shapes
:q
'freq_of_wrong_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)freq_of_wrong_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)freq_of_wrong_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!freq_of_wrong_words/strided_sliceStridedSlice"freq_of_wrong_words/Shape:output:00freq_of_wrong_words/strided_slice/stack:output:02freq_of_wrong_words/strided_slice/stack_1:output:02freq_of_wrong_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#freq_of_wrong_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!freq_of_wrong_words/Reshape/shapePack*freq_of_wrong_words/strided_slice:output:0,freq_of_wrong_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
freq_of_wrong_words/ReshapeReshape'freq_of_wrong_words/ExpandDims:output:0*freq_of_wrong_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#lexrank_avg_min_diff/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
lexrank_avg_min_diff/ExpandDims
ExpandDimsfeatures_lexrank_avg_min_diff,lexrank_avg_min_diff/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
lexrank_avg_min_diff/ShapeShape(lexrank_avg_min_diff/ExpandDims:output:0*
T0*
_output_shapes
:r
(lexrank_avg_min_diff/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*lexrank_avg_min_diff/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*lexrank_avg_min_diff/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"lexrank_avg_min_diff/strided_sliceStridedSlice#lexrank_avg_min_diff/Shape:output:01lexrank_avg_min_diff/strided_slice/stack:output:03lexrank_avg_min_diff/strided_slice/stack_1:output:03lexrank_avg_min_diff/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$lexrank_avg_min_diff/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"lexrank_avg_min_diff/Reshape/shapePack+lexrank_avg_min_diff/strided_slice:output:0-lexrank_avg_min_diff/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:°
lexrank_avg_min_diff/ReshapeReshape(lexrank_avg_min_diff/ExpandDims:output:0+lexrank_avg_min_diff/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙o
$lexrank_interquartile/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ż
 lexrank_interquartile/ExpandDims
ExpandDimsfeatures_lexrank_interquartile-lexrank_interquartile/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙t
lexrank_interquartile/ShapeShape)lexrank_interquartile/ExpandDims:output:0*
T0*
_output_shapes
:s
)lexrank_interquartile/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+lexrank_interquartile/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+lexrank_interquartile/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#lexrank_interquartile/strided_sliceStridedSlice$lexrank_interquartile/Shape:output:02lexrank_interquartile/strided_slice/stack:output:04lexrank_interquartile/strided_slice/stack_1:output:04lexrank_interquartile/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%lexrank_interquartile/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ˇ
#lexrank_interquartile/Reshape/shapePack,lexrank_interquartile/strided_slice:output:0.lexrank_interquartile/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ł
lexrank_interquartile/ReshapeReshape)lexrank_interquartile/ExpandDims:output:0,lexrank_interquartile/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙h
mcalpine_eflaw/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
mcalpine_eflaw/ExpandDims
ExpandDimsfeatures_mcalpine_eflaw&mcalpine_eflaw/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
mcalpine_eflaw/ShapeShape"mcalpine_eflaw/ExpandDims:output:0*
T0*
_output_shapes
:l
"mcalpine_eflaw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$mcalpine_eflaw/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$mcalpine_eflaw/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
mcalpine_eflaw/strided_sliceStridedSlicemcalpine_eflaw/Shape:output:0+mcalpine_eflaw/strided_slice/stack:output:0-mcalpine_eflaw/strided_slice/stack_1:output:0-mcalpine_eflaw/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask`
mcalpine_eflaw/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :˘
mcalpine_eflaw/Reshape/shapePack%mcalpine_eflaw/strided_slice:output:0'mcalpine_eflaw/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
mcalpine_eflaw/ReshapeReshape"mcalpine_eflaw/ExpandDims:output:0%mcalpine_eflaw/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
noun_to_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
noun_to_adj/ExpandDims
ExpandDimsfeatures_noun_to_adj#noun_to_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
noun_to_adj/ShapeShapenoun_to_adj/ExpandDims:output:0*
T0*
_output_shapes
:i
noun_to_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!noun_to_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!noun_to_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
noun_to_adj/strided_sliceStridedSlicenoun_to_adj/Shape:output:0(noun_to_adj/strided_slice/stack:output:0*noun_to_adj/strided_slice/stack_1:output:0*noun_to_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
noun_to_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
noun_to_adj/Reshape/shapePack"noun_to_adj/strided_slice:output:0$noun_to_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
noun_to_adj/ReshapeReshapenoun_to_adj/ExpandDims:output:0"noun_to_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙o
$num_of_grammar_errors/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ż
 num_of_grammar_errors/ExpandDims
ExpandDimsfeatures_num_of_grammar_errors-num_of_grammar_errors/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
num_of_grammar_errors/CastCast)num_of_grammar_errors/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
num_of_grammar_errors/ShapeShapenum_of_grammar_errors/Cast:y:0*
T0*
_output_shapes
:s
)num_of_grammar_errors/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+num_of_grammar_errors/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+num_of_grammar_errors/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ż
#num_of_grammar_errors/strided_sliceStridedSlice$num_of_grammar_errors/Shape:output:02num_of_grammar_errors/strided_slice/stack:output:04num_of_grammar_errors/strided_slice/stack_1:output:04num_of_grammar_errors/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%num_of_grammar_errors/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ˇ
#num_of_grammar_errors/Reshape/shapePack,num_of_grammar_errors/strided_slice:output:0.num_of_grammar_errors/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:¨
num_of_grammar_errors/ReshapeReshapenum_of_grammar_errors/Cast:y:0,num_of_grammar_errors/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!num_of_short_forms/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ś
num_of_short_forms/ExpandDims
ExpandDimsfeatures_num_of_short_forms*num_of_short_forms/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
num_of_short_forms/CastCast&num_of_short_forms/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙c
num_of_short_forms/ShapeShapenum_of_short_forms/Cast:y:0*
T0*
_output_shapes
:p
&num_of_short_forms/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(num_of_short_forms/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(num_of_short_forms/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 num_of_short_forms/strided_sliceStridedSlice!num_of_short_forms/Shape:output:0/num_of_short_forms/strided_slice/stack:output:01num_of_short_forms/strided_slice/stack_1:output:01num_of_short_forms/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"num_of_short_forms/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 num_of_short_forms/Reshape/shapePack)num_of_short_forms/strided_slice:output:0+num_of_short_forms/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
num_of_short_forms/ReshapeReshapenum_of_short_forms/Cast:y:0)num_of_short_forms/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
#number_of_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź
number_of_diff_words/ExpandDims
ExpandDimsfeatures_number_of_diff_words,number_of_diff_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
number_of_diff_words/CastCast(number_of_diff_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
number_of_diff_words/ShapeShapenumber_of_diff_words/Cast:y:0*
T0*
_output_shapes
:r
(number_of_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*number_of_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*number_of_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ş
"number_of_diff_words/strided_sliceStridedSlice#number_of_diff_words/Shape:output:01number_of_diff_words/strided_slice/stack:output:03number_of_diff_words/strided_slice/stack_1:output:03number_of_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$number_of_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :´
"number_of_diff_words/Reshape/shapePack+number_of_diff_words/strided_slice:output:0-number_of_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ľ
number_of_diff_words/ReshapeReshapenumber_of_diff_words/Cast:y:0+number_of_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙i
number_of_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
number_of_words/ExpandDims
ExpandDimsfeatures_number_of_words'number_of_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
number_of_words/CastCast#number_of_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙]
number_of_words/ShapeShapenumber_of_words/Cast:y:0*
T0*
_output_shapes
:m
#number_of_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%number_of_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%number_of_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ą
number_of_words/strided_sliceStridedSlicenumber_of_words/Shape:output:0,number_of_words/strided_slice/stack:output:0.number_of_words/strided_slice/stack_1:output:0.number_of_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
number_of_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ľ
number_of_words/Reshape/shapePack&number_of_words/strided_slice:output:0(number_of_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
number_of_words/ReshapeReshapenumber_of_words/Cast:y:0&number_of_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙j
phrase_diversity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ 
phrase_diversity/ExpandDims
ExpandDimsfeatures_phrase_diversity(phrase_diversity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙j
phrase_diversity/ShapeShape$phrase_diversity/ExpandDims:output:0*
T0*
_output_shapes
:n
$phrase_diversity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&phrase_diversity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&phrase_diversity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ś
phrase_diversity/strided_sliceStridedSlicephrase_diversity/Shape:output:0-phrase_diversity/strided_slice/stack:output:0/phrase_diversity/strided_slice/stack_1:output:0/phrase_diversity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 phrase_diversity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :¨
phrase_diversity/Reshape/shapePack'phrase_diversity/strided_slice:output:0)phrase_diversity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:¤
phrase_diversity/ReshapeReshape$phrase_diversity/ExpandDims:output:0'phrase_diversity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
punctuations/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
punctuations/ExpandDims
ExpandDimsfeatures_punctuations$punctuations/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙b
punctuations/ShapeShape punctuations/ExpandDims:output:0*
T0*
_output_shapes
:j
 punctuations/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"punctuations/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"punctuations/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
punctuations/strided_sliceStridedSlicepunctuations/Shape:output:0)punctuations/strided_slice/stack:output:0+punctuations/strided_slice/stack_1:output:0+punctuations/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
punctuations/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
punctuations/Reshape/shapePack#punctuations/strided_slice:output:0%punctuations/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
punctuations/ReshapeReshape punctuations/ExpandDims:output:0#punctuations/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"sentence_complexity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Š
sentence_complexity/ExpandDims
ExpandDimsfeatures_sentence_complexity+sentence_complexity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
sentence_complexity/ShapeShape'sentence_complexity/ExpandDims:output:0*
T0*
_output_shapes
:q
'sentence_complexity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sentence_complexity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sentence_complexity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!sentence_complexity/strided_sliceStridedSlice"sentence_complexity/Shape:output:00sentence_complexity/strided_slice/stack:output:02sentence_complexity/strided_slice/stack_1:output:02sentence_complexity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#sentence_complexity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!sentence_complexity/Reshape/shapePack*sentence_complexity/strided_slice:output:0,sentence_complexity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
sentence_complexity/ReshapeReshape'sentence_complexity/ExpandDims:output:0*sentence_complexity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!sentiment_compound/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ś
sentiment_compound/ExpandDims
ExpandDimsfeatures_sentiment_compound*sentiment_compound/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
sentiment_compound/ShapeShape&sentiment_compound/ExpandDims:output:0*
T0*
_output_shapes
:p
&sentiment_compound/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sentiment_compound/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sentiment_compound/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sentiment_compound/strided_sliceStridedSlice!sentiment_compound/Shape:output:0/sentiment_compound/strided_slice/stack:output:01sentiment_compound/strided_slice/stack_1:output:01sentiment_compound/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sentiment_compound/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 sentiment_compound/Reshape/shapePack)sentiment_compound/strided_slice:output:0+sentiment_compound/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
sentiment_compound/ReshapeReshape&sentiment_compound/ExpandDims:output:0)sentiment_compound/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!sentiment_negative/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ś
sentiment_negative/ExpandDims
ExpandDimsfeatures_sentiment_negative*sentiment_negative/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
sentiment_negative/ShapeShape&sentiment_negative/ExpandDims:output:0*
T0*
_output_shapes
:p
&sentiment_negative/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sentiment_negative/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sentiment_negative/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sentiment_negative/strided_sliceStridedSlice!sentiment_negative/Shape:output:0/sentiment_negative/strided_slice/stack:output:01sentiment_negative/strided_slice/stack_1:output:01sentiment_negative/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sentiment_negative/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 sentiment_negative/Reshape/shapePack)sentiment_negative/strided_slice:output:0+sentiment_negative/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
sentiment_negative/ReshapeReshape&sentiment_negative/ExpandDims:output:0)sentiment_negative/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!sentiment_positive/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ś
sentiment_positive/ExpandDims
ExpandDimsfeatures_sentiment_positive*sentiment_positive/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
sentiment_positive/ShapeShape&sentiment_positive/ExpandDims:output:0*
T0*
_output_shapes
:p
&sentiment_positive/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(sentiment_positive/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(sentiment_positive/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 sentiment_positive/strided_sliceStridedSlice!sentiment_positive/Shape:output:0/sentiment_positive/strided_slice/stack:output:01sentiment_positive/strided_slice/stack_1:output:01sentiment_positive/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sentiment_positive/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 sentiment_positive/Reshape/shapePack)sentiment_positive/strided_slice:output:0+sentiment_positive/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
sentiment_positive/ReshapeReshape&sentiment_positive/ExpandDims:output:0)sentiment_positive/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙m
"stopwords_frequency/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Š
stopwords_frequency/ExpandDims
ExpandDimsfeatures_stopwords_frequency+stopwords_frequency/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙p
stopwords_frequency/ShapeShape'stopwords_frequency/ExpandDims:output:0*
T0*
_output_shapes
:q
'stopwords_frequency/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)stopwords_frequency/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)stopwords_frequency/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ľ
!stopwords_frequency/strided_sliceStridedSlice"stopwords_frequency/Shape:output:00stopwords_frequency/strided_slice/stack:output:02stopwords_frequency/strided_slice/stack_1:output:02stopwords_frequency/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#stopwords_frequency/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ą
!stopwords_frequency/Reshape/shapePack*stopwords_frequency/strided_slice:output:0,stopwords_frequency/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:­
stopwords_frequency/ReshapeReshape'stopwords_frequency/ExpandDims:output:0*stopwords_frequency/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙g
text_standard/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
text_standard/ExpandDims
ExpandDimsfeatures_text_standard%text_standard/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙d
text_standard/ShapeShape!text_standard/ExpandDims:output:0*
T0*
_output_shapes
:k
!text_standard/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#text_standard/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#text_standard/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
text_standard/strided_sliceStridedSlicetext_standard/Shape:output:0*text_standard/strided_slice/stack:output:0,text_standard/strided_slice/stack_1:output:0,text_standard/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
text_standard/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
text_standard/Reshape/shapePack$text_standard/strided_slice:output:0&text_standard/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
text_standard/ReshapeReshape!text_standard/ExpandDims:output:0$text_standard/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙]
ttr/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙y
ttr/ExpandDims
ExpandDimsfeatures_ttrttr/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P
	ttr/ShapeShapettr/ExpandDims:output:0*
T0*
_output_shapes
:a
ttr/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
ttr/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
ttr/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ĺ
ttr/strided_sliceStridedSlicettr/Shape:output:0 ttr/strided_slice/stack:output:0"ttr/strided_slice/stack_1:output:0"ttr/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
ttr/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
ttr/Reshape/shapePackttr/strided_slice:output:0ttr/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:}
ttr/ReshapeReshapettr/ExpandDims:output:0ttr/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
verb_to_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
verb_to_adv/ExpandDims
ExpandDimsfeatures_verb_to_adv#verb_to_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
verb_to_adv/ShapeShapeverb_to_adv/ExpandDims:output:0*
T0*
_output_shapes
:i
verb_to_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!verb_to_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!verb_to_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
verb_to_adv/strided_sliceStridedSliceverb_to_adv/Shape:output:0(verb_to_adv/strided_slice/stack:output:0*verb_to_adv/strided_slice/stack_1:output:0*verb_to_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
verb_to_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :
verb_to_adv/Reshape/shapePack"verb_to_adv/strided_slice:output:0$verb_to_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
verb_to_adv/ReshapeReshapeverb_to_adv/ExpandDims:output:0"verb_to_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ź

concatConcatV2ARI/Reshape:output:0%Incorrect_form_ratio/Reshape:output:0 av_word_per_sen/Reshape:output:0 coherence_score/Reshape:output:0-dale_chall_readability_score/Reshape:output:0%flesch_kincaid_grade/Reshape:output:0$flesch_reading_ease/Reshape:output:0 freq_diff_words/Reshape:output:0freq_of_adj/Reshape:output:0freq_of_adv/Reshape:output:0%freq_of_distinct_adj/Reshape:output:0%freq_of_distinct_adv/Reshape:output:0freq_of_noun/Reshape:output:0 freq_of_pronoun/Reshape:output:0#freq_of_transition/Reshape:output:0freq_of_verb/Reshape:output:0$freq_of_wrong_words/Reshape:output:0%lexrank_avg_min_diff/Reshape:output:0&lexrank_interquartile/Reshape:output:0mcalpine_eflaw/Reshape:output:0noun_to_adj/Reshape:output:0&num_of_grammar_errors/Reshape:output:0#num_of_short_forms/Reshape:output:0%number_of_diff_words/Reshape:output:0 number_of_words/Reshape:output:0!phrase_diversity/Reshape:output:0punctuations/Reshape:output:0$sentence_complexity/Reshape:output:0#sentiment_compound/Reshape:output:0#sentiment_negative/Reshape:output:0#sentiment_positive/Reshape:output:0$stopwords_frequency/Reshape:output:0text_standard/Reshape:output:0ttr/Reshape:output:0verb_to_adv/Reshape:output:0concat/axis:output:0*
N#*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙#W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙#"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ŕ
_input_shapesŽ
Ť:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:Q M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefeatures/ARI:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/Incorrect_form_ratio:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/av_word_per_sen:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/coherence_score:VR
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
+
_user_specified_namefeatures/cohesion:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_namefeatures/corrected_text:jf
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
?
_user_specified_name'%features/dale_chall_readability_score:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/flesch_kincaid_grade:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/flesch_reading_ease:]	Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/freq_diff_words:Y
U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/freq_of_adj:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/freq_of_adv:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/freq_of_distinct_adj:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/freq_of_distinct_adv:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namefeatures/freq_of_noun:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/freq_of_pronoun:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/freq_of_transition:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namefeatures/freq_of_verb:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/freq_of_wrong_words:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/lexrank_avg_min_diff:c_
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name features/lexrank_interquartile:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_namefeatures/mcalpine_eflaw:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/noun_to_adj:c_
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
8
_user_specified_name features/num_of_grammar_errors:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/num_of_short_forms:b^
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
7
_user_specified_namefeatures/number_of_diff_words:]Y
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
_user_specified_namefeatures/number_of_words:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_namefeatures/phrase_diversity:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namefeatures/punctuations:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/sentence_complexity:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/sentiment_compound:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/sentiment_negative:` \
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_namefeatures/sentiment_positive:a!]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namefeatures/stopwords_frequency:["W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_namefeatures/text_standard:Q#M
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefeatures/ttr:Y$U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefeatures/verb_to_adv
Đś

G__inference_sequential_2_layer_call_and_return_conditional_losses_52975

inputs_ari	
inputs_incorrect_form_ratio
inputs_av_word_per_sen
inputs_coherence_score
inputs_cohesion
inputs_corrected_text'
#inputs_dale_chall_readability_score
inputs_flesch_kincaid_grade
inputs_flesch_reading_ease
inputs_freq_diff_words
inputs_freq_of_adj
inputs_freq_of_adv
inputs_freq_of_distinct_adj
inputs_freq_of_distinct_adv
inputs_freq_of_noun
inputs_freq_of_pronoun
inputs_freq_of_transition
inputs_freq_of_verb
inputs_freq_of_wrong_words
inputs_lexrank_avg_min_diff 
inputs_lexrank_interquartile
inputs_mcalpine_eflaw
inputs_noun_to_adj 
inputs_num_of_grammar_errors	
inputs_num_of_short_forms	
inputs_number_of_diff_words	
inputs_number_of_words	
inputs_phrase_diversity
inputs_punctuations
inputs_sentence_complexity
inputs_sentiment_compound
inputs_sentiment_negative
inputs_sentiment_positive
inputs_stopwords_frequency
inputs_text_standard

inputs_ttr
inputs_verb_to_adv8
&hidden0_matmul_readvariableop_resource:#5
'hidden0_biasadd_readvariableop_resource:8
&hidden1_matmul_readvariableop_resource:5
'hidden1_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity˘Hidden0/BiasAdd/ReadVariableOp˘Hidden0/MatMul/ReadVariableOp˘Hidden1/BiasAdd/ReadVariableOp˘Hidden1/MatMul/ReadVariableOp˘Output/BiasAdd/ReadVariableOp˘Output/MatMul/ReadVariableOpl
!dense_features/ARI/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
dense_features/ARI/ExpandDims
ExpandDims
inputs_ari*dense_features/ARI/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_features/ARI/CastCast&dense_features/ARI/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙c
dense_features/ARI/ShapeShapedense_features/ARI/Cast:y:0*
T0*
_output_shapes
:p
&dense_features/ARI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/ARI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/ARI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 dense_features/ARI/strided_sliceStridedSlice!dense_features/ARI/Shape:output:0/dense_features/ARI/strided_slice/stack:output:01dense_features/ARI/strided_slice/stack_1:output:01dense_features/ARI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/ARI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 dense_features/ARI/Reshape/shapePack)dense_features/ARI/strided_slice:output:0+dense_features/ARI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
dense_features/ARI/ReshapeReshapedense_features/ARI/Cast:y:0)dense_features/ARI/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙}
2dense_features/Incorrect_form_ratio/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Č
.dense_features/Incorrect_form_ratio/ExpandDims
ExpandDimsinputs_incorrect_form_ratio;dense_features/Incorrect_form_ratio/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)dense_features/Incorrect_form_ratio/ShapeShape7dense_features/Incorrect_form_ratio/ExpandDims:output:0*
T0*
_output_shapes
:
7dense_features/Incorrect_form_ratio/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9dense_features/Incorrect_form_ratio/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9dense_features/Incorrect_form_ratio/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1dense_features/Incorrect_form_ratio/strided_sliceStridedSlice2dense_features/Incorrect_form_ratio/Shape:output:0@dense_features/Incorrect_form_ratio/strided_slice/stack:output:0Bdense_features/Incorrect_form_ratio/strided_slice/stack_1:output:0Bdense_features/Incorrect_form_ratio/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/Incorrect_form_ratio/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :á
1dense_features/Incorrect_form_ratio/Reshape/shapePack:dense_features/Incorrect_form_ratio/strided_slice:output:0<dense_features/Incorrect_form_ratio/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ý
+dense_features/Incorrect_form_ratio/ReshapeReshape7dense_features/Incorrect_form_ratio/ExpandDims:output:0:dense_features/Incorrect_form_ratio/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙x
-dense_features/av_word_per_sen/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙š
)dense_features/av_word_per_sen/ExpandDims
ExpandDimsinputs_av_word_per_sen6dense_features/av_word_per_sen/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$dense_features/av_word_per_sen/ShapeShape2dense_features/av_word_per_sen/ExpandDims:output:0*
T0*
_output_shapes
:|
2dense_features/av_word_per_sen/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/av_word_per_sen/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/av_word_per_sen/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
,dense_features/av_word_per_sen/strided_sliceStridedSlice-dense_features/av_word_per_sen/Shape:output:0;dense_features/av_word_per_sen/strided_slice/stack:output:0=dense_features/av_word_per_sen/strided_slice/stack_1:output:0=dense_features/av_word_per_sen/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/av_word_per_sen/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ň
,dense_features/av_word_per_sen/Reshape/shapePack5dense_features/av_word_per_sen/strided_slice:output:07dense_features/av_word_per_sen/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Î
&dense_features/av_word_per_sen/ReshapeReshape2dense_features/av_word_per_sen/ExpandDims:output:05dense_features/av_word_per_sen/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙x
-dense_features/coherence_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙š
)dense_features/coherence_score/ExpandDims
ExpandDimsinputs_coherence_score6dense_features/coherence_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$dense_features/coherence_score/ShapeShape2dense_features/coherence_score/ExpandDims:output:0*
T0*
_output_shapes
:|
2dense_features/coherence_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/coherence_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/coherence_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
,dense_features/coherence_score/strided_sliceStridedSlice-dense_features/coherence_score/Shape:output:0;dense_features/coherence_score/strided_slice/stack:output:0=dense_features/coherence_score/strided_slice/stack_1:output:0=dense_features/coherence_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/coherence_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ň
,dense_features/coherence_score/Reshape/shapePack5dense_features/coherence_score/strided_slice:output:07dense_features/coherence_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Î
&dense_features/coherence_score/ReshapeReshape2dense_features/coherence_score/ExpandDims:output:05dense_features/coherence_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
:dense_features/dale_chall_readability_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ŕ
6dense_features/dale_chall_readability_score/ExpandDims
ExpandDims#inputs_dale_chall_readability_scoreCdense_features/dale_chall_readability_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
1dense_features/dale_chall_readability_score/ShapeShape?dense_features/dale_chall_readability_score/ExpandDims:output:0*
T0*
_output_shapes
:
?dense_features/dale_chall_readability_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Adense_features/dale_chall_readability_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Adense_features/dale_chall_readability_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
9dense_features/dale_chall_readability_score/strided_sliceStridedSlice:dense_features/dale_chall_readability_score/Shape:output:0Hdense_features/dale_chall_readability_score/strided_slice/stack:output:0Jdense_features/dale_chall_readability_score/strided_slice/stack_1:output:0Jdense_features/dale_chall_readability_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;dense_features/dale_chall_readability_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ů
9dense_features/dale_chall_readability_score/Reshape/shapePackBdense_features/dale_chall_readability_score/strided_slice:output:0Ddense_features/dale_chall_readability_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ő
3dense_features/dale_chall_readability_score/ReshapeReshape?dense_features/dale_chall_readability_score/ExpandDims:output:0Bdense_features/dale_chall_readability_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙}
2dense_features/flesch_kincaid_grade/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Č
.dense_features/flesch_kincaid_grade/ExpandDims
ExpandDimsinputs_flesch_kincaid_grade;dense_features/flesch_kincaid_grade/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)dense_features/flesch_kincaid_grade/ShapeShape7dense_features/flesch_kincaid_grade/ExpandDims:output:0*
T0*
_output_shapes
:
7dense_features/flesch_kincaid_grade/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9dense_features/flesch_kincaid_grade/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9dense_features/flesch_kincaid_grade/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1dense_features/flesch_kincaid_grade/strided_sliceStridedSlice2dense_features/flesch_kincaid_grade/Shape:output:0@dense_features/flesch_kincaid_grade/strided_slice/stack:output:0Bdense_features/flesch_kincaid_grade/strided_slice/stack_1:output:0Bdense_features/flesch_kincaid_grade/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/flesch_kincaid_grade/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :á
1dense_features/flesch_kincaid_grade/Reshape/shapePack:dense_features/flesch_kincaid_grade/strided_slice:output:0<dense_features/flesch_kincaid_grade/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ý
+dense_features/flesch_kincaid_grade/ReshapeReshape7dense_features/flesch_kincaid_grade/ExpandDims:output:0:dense_features/flesch_kincaid_grade/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙|
1dense_features/flesch_reading_ease/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ĺ
-dense_features/flesch_reading_ease/ExpandDims
ExpandDimsinputs_flesch_reading_ease:dense_features/flesch_reading_ease/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(dense_features/flesch_reading_ease/ShapeShape6dense_features/flesch_reading_ease/ExpandDims:output:0*
T0*
_output_shapes
:
6dense_features/flesch_reading_ease/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8dense_features/flesch_reading_ease/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8dense_features/flesch_reading_ease/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0dense_features/flesch_reading_ease/strided_sliceStridedSlice1dense_features/flesch_reading_ease/Shape:output:0?dense_features/flesch_reading_ease/strided_slice/stack:output:0Adense_features/flesch_reading_ease/strided_slice/stack_1:output:0Adense_features/flesch_reading_ease/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2dense_features/flesch_reading_ease/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ţ
0dense_features/flesch_reading_ease/Reshape/shapePack9dense_features/flesch_reading_ease/strided_slice:output:0;dense_features/flesch_reading_ease/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ú
*dense_features/flesch_reading_ease/ReshapeReshape6dense_features/flesch_reading_ease/ExpandDims:output:09dense_features/flesch_reading_ease/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙x
-dense_features/freq_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙š
)dense_features/freq_diff_words/ExpandDims
ExpandDimsinputs_freq_diff_words6dense_features/freq_diff_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$dense_features/freq_diff_words/ShapeShape2dense_features/freq_diff_words/ExpandDims:output:0*
T0*
_output_shapes
:|
2dense_features/freq_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/freq_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/freq_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
,dense_features/freq_diff_words/strided_sliceStridedSlice-dense_features/freq_diff_words/Shape:output:0;dense_features/freq_diff_words/strided_slice/stack:output:0=dense_features/freq_diff_words/strided_slice/stack_1:output:0=dense_features/freq_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/freq_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ň
,dense_features/freq_diff_words/Reshape/shapePack5dense_features/freq_diff_words/strided_slice:output:07dense_features/freq_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Î
&dense_features/freq_diff_words/ReshapeReshape2dense_features/freq_diff_words/ExpandDims:output:05dense_features/freq_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙t
)dense_features/freq_of_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙­
%dense_features/freq_of_adj/ExpandDims
ExpandDimsinputs_freq_of_adj2dense_features/freq_of_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
 dense_features/freq_of_adj/ShapeShape.dense_features/freq_of_adj/ExpandDims:output:0*
T0*
_output_shapes
:x
.dense_features/freq_of_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0dense_features/freq_of_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0dense_features/freq_of_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(dense_features/freq_of_adj/strided_sliceStridedSlice)dense_features/freq_of_adj/Shape:output:07dense_features/freq_of_adj/strided_slice/stack:output:09dense_features/freq_of_adj/strided_slice/stack_1:output:09dense_features/freq_of_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*dense_features/freq_of_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ć
(dense_features/freq_of_adj/Reshape/shapePack1dense_features/freq_of_adj/strided_slice:output:03dense_features/freq_of_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Â
"dense_features/freq_of_adj/ReshapeReshape.dense_features/freq_of_adj/ExpandDims:output:01dense_features/freq_of_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙t
)dense_features/freq_of_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙­
%dense_features/freq_of_adv/ExpandDims
ExpandDimsinputs_freq_of_adv2dense_features/freq_of_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
 dense_features/freq_of_adv/ShapeShape.dense_features/freq_of_adv/ExpandDims:output:0*
T0*
_output_shapes
:x
.dense_features/freq_of_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0dense_features/freq_of_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0dense_features/freq_of_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(dense_features/freq_of_adv/strided_sliceStridedSlice)dense_features/freq_of_adv/Shape:output:07dense_features/freq_of_adv/strided_slice/stack:output:09dense_features/freq_of_adv/strided_slice/stack_1:output:09dense_features/freq_of_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*dense_features/freq_of_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ć
(dense_features/freq_of_adv/Reshape/shapePack1dense_features/freq_of_adv/strided_slice:output:03dense_features/freq_of_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Â
"dense_features/freq_of_adv/ReshapeReshape.dense_features/freq_of_adv/ExpandDims:output:01dense_features/freq_of_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙}
2dense_features/freq_of_distinct_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Č
.dense_features/freq_of_distinct_adj/ExpandDims
ExpandDimsinputs_freq_of_distinct_adj;dense_features/freq_of_distinct_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)dense_features/freq_of_distinct_adj/ShapeShape7dense_features/freq_of_distinct_adj/ExpandDims:output:0*
T0*
_output_shapes
:
7dense_features/freq_of_distinct_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9dense_features/freq_of_distinct_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9dense_features/freq_of_distinct_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1dense_features/freq_of_distinct_adj/strided_sliceStridedSlice2dense_features/freq_of_distinct_adj/Shape:output:0@dense_features/freq_of_distinct_adj/strided_slice/stack:output:0Bdense_features/freq_of_distinct_adj/strided_slice/stack_1:output:0Bdense_features/freq_of_distinct_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/freq_of_distinct_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :á
1dense_features/freq_of_distinct_adj/Reshape/shapePack:dense_features/freq_of_distinct_adj/strided_slice:output:0<dense_features/freq_of_distinct_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ý
+dense_features/freq_of_distinct_adj/ReshapeReshape7dense_features/freq_of_distinct_adj/ExpandDims:output:0:dense_features/freq_of_distinct_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙}
2dense_features/freq_of_distinct_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Č
.dense_features/freq_of_distinct_adv/ExpandDims
ExpandDimsinputs_freq_of_distinct_adv;dense_features/freq_of_distinct_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)dense_features/freq_of_distinct_adv/ShapeShape7dense_features/freq_of_distinct_adv/ExpandDims:output:0*
T0*
_output_shapes
:
7dense_features/freq_of_distinct_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9dense_features/freq_of_distinct_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9dense_features/freq_of_distinct_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1dense_features/freq_of_distinct_adv/strided_sliceStridedSlice2dense_features/freq_of_distinct_adv/Shape:output:0@dense_features/freq_of_distinct_adv/strided_slice/stack:output:0Bdense_features/freq_of_distinct_adv/strided_slice/stack_1:output:0Bdense_features/freq_of_distinct_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/freq_of_distinct_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :á
1dense_features/freq_of_distinct_adv/Reshape/shapePack:dense_features/freq_of_distinct_adv/strided_slice:output:0<dense_features/freq_of_distinct_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ý
+dense_features/freq_of_distinct_adv/ReshapeReshape7dense_features/freq_of_distinct_adv/ExpandDims:output:0:dense_features/freq_of_distinct_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙u
*dense_features/freq_of_noun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙°
&dense_features/freq_of_noun/ExpandDims
ExpandDimsinputs_freq_of_noun3dense_features/freq_of_noun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!dense_features/freq_of_noun/ShapeShape/dense_features/freq_of_noun/ExpandDims:output:0*
T0*
_output_shapes
:y
/dense_features/freq_of_noun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/freq_of_noun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/freq_of_noun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)dense_features/freq_of_noun/strided_sliceStridedSlice*dense_features/freq_of_noun/Shape:output:08dense_features/freq_of_noun/strided_slice/stack:output:0:dense_features/freq_of_noun/strided_slice/stack_1:output:0:dense_features/freq_of_noun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/freq_of_noun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :É
)dense_features/freq_of_noun/Reshape/shapePack2dense_features/freq_of_noun/strided_slice:output:04dense_features/freq_of_noun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ĺ
#dense_features/freq_of_noun/ReshapeReshape/dense_features/freq_of_noun/ExpandDims:output:02dense_features/freq_of_noun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙x
-dense_features/freq_of_pronoun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙š
)dense_features/freq_of_pronoun/ExpandDims
ExpandDimsinputs_freq_of_pronoun6dense_features/freq_of_pronoun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$dense_features/freq_of_pronoun/ShapeShape2dense_features/freq_of_pronoun/ExpandDims:output:0*
T0*
_output_shapes
:|
2dense_features/freq_of_pronoun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/freq_of_pronoun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/freq_of_pronoun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
,dense_features/freq_of_pronoun/strided_sliceStridedSlice-dense_features/freq_of_pronoun/Shape:output:0;dense_features/freq_of_pronoun/strided_slice/stack:output:0=dense_features/freq_of_pronoun/strided_slice/stack_1:output:0=dense_features/freq_of_pronoun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/freq_of_pronoun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ň
,dense_features/freq_of_pronoun/Reshape/shapePack5dense_features/freq_of_pronoun/strided_slice:output:07dense_features/freq_of_pronoun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Î
&dense_features/freq_of_pronoun/ReshapeReshape2dense_features/freq_of_pronoun/ExpandDims:output:05dense_features/freq_of_pronoun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
0dense_features/freq_of_transition/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Â
,dense_features/freq_of_transition/ExpandDims
ExpandDimsinputs_freq_of_transition9dense_features/freq_of_transition/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'dense_features/freq_of_transition/ShapeShape5dense_features/freq_of_transition/ExpandDims:output:0*
T0*
_output_shapes
:
5dense_features/freq_of_transition/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7dense_features/freq_of_transition/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7dense_features/freq_of_transition/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ű
/dense_features/freq_of_transition/strided_sliceStridedSlice0dense_features/freq_of_transition/Shape:output:0>dense_features/freq_of_transition/strided_slice/stack:output:0@dense_features/freq_of_transition/strided_slice/stack_1:output:0@dense_features/freq_of_transition/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features/freq_of_transition/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ű
/dense_features/freq_of_transition/Reshape/shapePack8dense_features/freq_of_transition/strided_slice:output:0:dense_features/freq_of_transition/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:×
)dense_features/freq_of_transition/ReshapeReshape5dense_features/freq_of_transition/ExpandDims:output:08dense_features/freq_of_transition/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙u
*dense_features/freq_of_verb/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙°
&dense_features/freq_of_verb/ExpandDims
ExpandDimsinputs_freq_of_verb3dense_features/freq_of_verb/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!dense_features/freq_of_verb/ShapeShape/dense_features/freq_of_verb/ExpandDims:output:0*
T0*
_output_shapes
:y
/dense_features/freq_of_verb/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/freq_of_verb/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/freq_of_verb/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)dense_features/freq_of_verb/strided_sliceStridedSlice*dense_features/freq_of_verb/Shape:output:08dense_features/freq_of_verb/strided_slice/stack:output:0:dense_features/freq_of_verb/strided_slice/stack_1:output:0:dense_features/freq_of_verb/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/freq_of_verb/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :É
)dense_features/freq_of_verb/Reshape/shapePack2dense_features/freq_of_verb/strided_slice:output:04dense_features/freq_of_verb/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ĺ
#dense_features/freq_of_verb/ReshapeReshape/dense_features/freq_of_verb/ExpandDims:output:02dense_features/freq_of_verb/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙|
1dense_features/freq_of_wrong_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ĺ
-dense_features/freq_of_wrong_words/ExpandDims
ExpandDimsinputs_freq_of_wrong_words:dense_features/freq_of_wrong_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(dense_features/freq_of_wrong_words/ShapeShape6dense_features/freq_of_wrong_words/ExpandDims:output:0*
T0*
_output_shapes
:
6dense_features/freq_of_wrong_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8dense_features/freq_of_wrong_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8dense_features/freq_of_wrong_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0dense_features/freq_of_wrong_words/strided_sliceStridedSlice1dense_features/freq_of_wrong_words/Shape:output:0?dense_features/freq_of_wrong_words/strided_slice/stack:output:0Adense_features/freq_of_wrong_words/strided_slice/stack_1:output:0Adense_features/freq_of_wrong_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2dense_features/freq_of_wrong_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ţ
0dense_features/freq_of_wrong_words/Reshape/shapePack9dense_features/freq_of_wrong_words/strided_slice:output:0;dense_features/freq_of_wrong_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ú
*dense_features/freq_of_wrong_words/ReshapeReshape6dense_features/freq_of_wrong_words/ExpandDims:output:09dense_features/freq_of_wrong_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙}
2dense_features/lexrank_avg_min_diff/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Č
.dense_features/lexrank_avg_min_diff/ExpandDims
ExpandDimsinputs_lexrank_avg_min_diff;dense_features/lexrank_avg_min_diff/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)dense_features/lexrank_avg_min_diff/ShapeShape7dense_features/lexrank_avg_min_diff/ExpandDims:output:0*
T0*
_output_shapes
:
7dense_features/lexrank_avg_min_diff/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9dense_features/lexrank_avg_min_diff/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9dense_features/lexrank_avg_min_diff/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1dense_features/lexrank_avg_min_diff/strided_sliceStridedSlice2dense_features/lexrank_avg_min_diff/Shape:output:0@dense_features/lexrank_avg_min_diff/strided_slice/stack:output:0Bdense_features/lexrank_avg_min_diff/strided_slice/stack_1:output:0Bdense_features/lexrank_avg_min_diff/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/lexrank_avg_min_diff/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :á
1dense_features/lexrank_avg_min_diff/Reshape/shapePack:dense_features/lexrank_avg_min_diff/strided_slice:output:0<dense_features/lexrank_avg_min_diff/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ý
+dense_features/lexrank_avg_min_diff/ReshapeReshape7dense_features/lexrank_avg_min_diff/ExpandDims:output:0:dense_features/lexrank_avg_min_diff/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
3dense_features/lexrank_interquartile/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ë
/dense_features/lexrank_interquartile/ExpandDims
ExpandDimsinputs_lexrank_interquartile<dense_features/lexrank_interquartile/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*dense_features/lexrank_interquartile/ShapeShape8dense_features/lexrank_interquartile/ExpandDims:output:0*
T0*
_output_shapes
:
8dense_features/lexrank_interquartile/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:dense_features/lexrank_interquartile/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:dense_features/lexrank_interquartile/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2dense_features/lexrank_interquartile/strided_sliceStridedSlice3dense_features/lexrank_interquartile/Shape:output:0Adense_features/lexrank_interquartile/strided_slice/stack:output:0Cdense_features/lexrank_interquartile/strided_slice/stack_1:output:0Cdense_features/lexrank_interquartile/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/lexrank_interquartile/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ä
2dense_features/lexrank_interquartile/Reshape/shapePack;dense_features/lexrank_interquartile/strided_slice:output:0=dense_features/lexrank_interquartile/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ŕ
,dense_features/lexrank_interquartile/ReshapeReshape8dense_features/lexrank_interquartile/ExpandDims:output:0;dense_features/lexrank_interquartile/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
,dense_features/mcalpine_eflaw/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ś
(dense_features/mcalpine_eflaw/ExpandDims
ExpandDimsinputs_mcalpine_eflaw5dense_features/mcalpine_eflaw/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#dense_features/mcalpine_eflaw/ShapeShape1dense_features/mcalpine_eflaw/ExpandDims:output:0*
T0*
_output_shapes
:{
1dense_features/mcalpine_eflaw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_features/mcalpine_eflaw/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_features/mcalpine_eflaw/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+dense_features/mcalpine_eflaw/strided_sliceStridedSlice,dense_features/mcalpine_eflaw/Shape:output:0:dense_features/mcalpine_eflaw/strided_slice/stack:output:0<dense_features/mcalpine_eflaw/strided_slice/stack_1:output:0<dense_features/mcalpine_eflaw/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-dense_features/mcalpine_eflaw/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ď
+dense_features/mcalpine_eflaw/Reshape/shapePack4dense_features/mcalpine_eflaw/strided_slice:output:06dense_features/mcalpine_eflaw/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ë
%dense_features/mcalpine_eflaw/ReshapeReshape1dense_features/mcalpine_eflaw/ExpandDims:output:04dense_features/mcalpine_eflaw/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙t
)dense_features/noun_to_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙­
%dense_features/noun_to_adj/ExpandDims
ExpandDimsinputs_noun_to_adj2dense_features/noun_to_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
 dense_features/noun_to_adj/ShapeShape.dense_features/noun_to_adj/ExpandDims:output:0*
T0*
_output_shapes
:x
.dense_features/noun_to_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0dense_features/noun_to_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0dense_features/noun_to_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(dense_features/noun_to_adj/strided_sliceStridedSlice)dense_features/noun_to_adj/Shape:output:07dense_features/noun_to_adj/strided_slice/stack:output:09dense_features/noun_to_adj/strided_slice/stack_1:output:09dense_features/noun_to_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*dense_features/noun_to_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ć
(dense_features/noun_to_adj/Reshape/shapePack1dense_features/noun_to_adj/strided_slice:output:03dense_features/noun_to_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Â
"dense_features/noun_to_adj/ReshapeReshape.dense_features/noun_to_adj/ExpandDims:output:01dense_features/noun_to_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
3dense_features/num_of_grammar_errors/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ë
/dense_features/num_of_grammar_errors/ExpandDims
ExpandDimsinputs_num_of_grammar_errors<dense_features/num_of_grammar_errors/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
)dense_features/num_of_grammar_errors/CastCast8dense_features/num_of_grammar_errors/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*dense_features/num_of_grammar_errors/ShapeShape-dense_features/num_of_grammar_errors/Cast:y:0*
T0*
_output_shapes
:
8dense_features/num_of_grammar_errors/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:dense_features/num_of_grammar_errors/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:dense_features/num_of_grammar_errors/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2dense_features/num_of_grammar_errors/strided_sliceStridedSlice3dense_features/num_of_grammar_errors/Shape:output:0Adense_features/num_of_grammar_errors/strided_slice/stack:output:0Cdense_features/num_of_grammar_errors/strided_slice/stack_1:output:0Cdense_features/num_of_grammar_errors/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/num_of_grammar_errors/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ä
2dense_features/num_of_grammar_errors/Reshape/shapePack;dense_features/num_of_grammar_errors/strided_slice:output:0=dense_features/num_of_grammar_errors/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ő
,dense_features/num_of_grammar_errors/ReshapeReshape-dense_features/num_of_grammar_errors/Cast:y:0;dense_features/num_of_grammar_errors/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
0dense_features/num_of_short_forms/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Â
,dense_features/num_of_short_forms/ExpandDims
ExpandDimsinputs_num_of_short_forms9dense_features/num_of_short_forms/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
&dense_features/num_of_short_forms/CastCast5dense_features/num_of_short_forms/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'dense_features/num_of_short_forms/ShapeShape*dense_features/num_of_short_forms/Cast:y:0*
T0*
_output_shapes
:
5dense_features/num_of_short_forms/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7dense_features/num_of_short_forms/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7dense_features/num_of_short_forms/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ű
/dense_features/num_of_short_forms/strided_sliceStridedSlice0dense_features/num_of_short_forms/Shape:output:0>dense_features/num_of_short_forms/strided_slice/stack:output:0@dense_features/num_of_short_forms/strided_slice/stack_1:output:0@dense_features/num_of_short_forms/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features/num_of_short_forms/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ű
/dense_features/num_of_short_forms/Reshape/shapePack8dense_features/num_of_short_forms/strided_slice:output:0:dense_features/num_of_short_forms/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ě
)dense_features/num_of_short_forms/ReshapeReshape*dense_features/num_of_short_forms/Cast:y:08dense_features/num_of_short_forms/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙}
2dense_features/number_of_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Č
.dense_features/number_of_diff_words/ExpandDims
ExpandDimsinputs_number_of_diff_words;dense_features/number_of_diff_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
(dense_features/number_of_diff_words/CastCast7dense_features/number_of_diff_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)dense_features/number_of_diff_words/ShapeShape,dense_features/number_of_diff_words/Cast:y:0*
T0*
_output_shapes
:
7dense_features/number_of_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9dense_features/number_of_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9dense_features/number_of_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1dense_features/number_of_diff_words/strided_sliceStridedSlice2dense_features/number_of_diff_words/Shape:output:0@dense_features/number_of_diff_words/strided_slice/stack:output:0Bdense_features/number_of_diff_words/strided_slice/stack_1:output:0Bdense_features/number_of_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/number_of_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :á
1dense_features/number_of_diff_words/Reshape/shapePack:dense_features/number_of_diff_words/strided_slice:output:0<dense_features/number_of_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ň
+dense_features/number_of_diff_words/ReshapeReshape,dense_features/number_of_diff_words/Cast:y:0:dense_features/number_of_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙x
-dense_features/number_of_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙š
)dense_features/number_of_words/ExpandDims
ExpandDimsinputs_number_of_words6dense_features/number_of_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
#dense_features/number_of_words/CastCast2dense_features/number_of_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
$dense_features/number_of_words/ShapeShape'dense_features/number_of_words/Cast:y:0*
T0*
_output_shapes
:|
2dense_features/number_of_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/number_of_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/number_of_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
,dense_features/number_of_words/strided_sliceStridedSlice-dense_features/number_of_words/Shape:output:0;dense_features/number_of_words/strided_slice/stack:output:0=dense_features/number_of_words/strided_slice/stack_1:output:0=dense_features/number_of_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/number_of_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ň
,dense_features/number_of_words/Reshape/shapePack5dense_features/number_of_words/strided_slice:output:07dense_features/number_of_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ă
&dense_features/number_of_words/ReshapeReshape'dense_features/number_of_words/Cast:y:05dense_features/number_of_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙y
.dense_features/phrase_diversity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ź
*dense_features/phrase_diversity/ExpandDims
ExpandDimsinputs_phrase_diversity7dense_features/phrase_diversity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
%dense_features/phrase_diversity/ShapeShape3dense_features/phrase_diversity/ExpandDims:output:0*
T0*
_output_shapes
:}
3dense_features/phrase_diversity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_features/phrase_diversity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_features/phrase_diversity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ń
-dense_features/phrase_diversity/strided_sliceStridedSlice.dense_features/phrase_diversity/Shape:output:0<dense_features/phrase_diversity/strided_slice/stack:output:0>dense_features/phrase_diversity/strided_slice/stack_1:output:0>dense_features/phrase_diversity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/dense_features/phrase_diversity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ő
-dense_features/phrase_diversity/Reshape/shapePack6dense_features/phrase_diversity/strided_slice:output:08dense_features/phrase_diversity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ń
'dense_features/phrase_diversity/ReshapeReshape3dense_features/phrase_diversity/ExpandDims:output:06dense_features/phrase_diversity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙u
*dense_features/punctuations/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙°
&dense_features/punctuations/ExpandDims
ExpandDimsinputs_punctuations3dense_features/punctuations/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!dense_features/punctuations/ShapeShape/dense_features/punctuations/ExpandDims:output:0*
T0*
_output_shapes
:y
/dense_features/punctuations/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/punctuations/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/punctuations/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)dense_features/punctuations/strided_sliceStridedSlice*dense_features/punctuations/Shape:output:08dense_features/punctuations/strided_slice/stack:output:0:dense_features/punctuations/strided_slice/stack_1:output:0:dense_features/punctuations/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/punctuations/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :É
)dense_features/punctuations/Reshape/shapePack2dense_features/punctuations/strided_slice:output:04dense_features/punctuations/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ĺ
#dense_features/punctuations/ReshapeReshape/dense_features/punctuations/ExpandDims:output:02dense_features/punctuations/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙|
1dense_features/sentence_complexity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ĺ
-dense_features/sentence_complexity/ExpandDims
ExpandDimsinputs_sentence_complexity:dense_features/sentence_complexity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(dense_features/sentence_complexity/ShapeShape6dense_features/sentence_complexity/ExpandDims:output:0*
T0*
_output_shapes
:
6dense_features/sentence_complexity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8dense_features/sentence_complexity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8dense_features/sentence_complexity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0dense_features/sentence_complexity/strided_sliceStridedSlice1dense_features/sentence_complexity/Shape:output:0?dense_features/sentence_complexity/strided_slice/stack:output:0Adense_features/sentence_complexity/strided_slice/stack_1:output:0Adense_features/sentence_complexity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2dense_features/sentence_complexity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ţ
0dense_features/sentence_complexity/Reshape/shapePack9dense_features/sentence_complexity/strided_slice:output:0;dense_features/sentence_complexity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ú
*dense_features/sentence_complexity/ReshapeReshape6dense_features/sentence_complexity/ExpandDims:output:09dense_features/sentence_complexity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
0dense_features/sentiment_compound/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Â
,dense_features/sentiment_compound/ExpandDims
ExpandDimsinputs_sentiment_compound9dense_features/sentiment_compound/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'dense_features/sentiment_compound/ShapeShape5dense_features/sentiment_compound/ExpandDims:output:0*
T0*
_output_shapes
:
5dense_features/sentiment_compound/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7dense_features/sentiment_compound/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7dense_features/sentiment_compound/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ű
/dense_features/sentiment_compound/strided_sliceStridedSlice0dense_features/sentiment_compound/Shape:output:0>dense_features/sentiment_compound/strided_slice/stack:output:0@dense_features/sentiment_compound/strided_slice/stack_1:output:0@dense_features/sentiment_compound/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features/sentiment_compound/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ű
/dense_features/sentiment_compound/Reshape/shapePack8dense_features/sentiment_compound/strided_slice:output:0:dense_features/sentiment_compound/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:×
)dense_features/sentiment_compound/ReshapeReshape5dense_features/sentiment_compound/ExpandDims:output:08dense_features/sentiment_compound/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
0dense_features/sentiment_negative/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Â
,dense_features/sentiment_negative/ExpandDims
ExpandDimsinputs_sentiment_negative9dense_features/sentiment_negative/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'dense_features/sentiment_negative/ShapeShape5dense_features/sentiment_negative/ExpandDims:output:0*
T0*
_output_shapes
:
5dense_features/sentiment_negative/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7dense_features/sentiment_negative/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7dense_features/sentiment_negative/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ű
/dense_features/sentiment_negative/strided_sliceStridedSlice0dense_features/sentiment_negative/Shape:output:0>dense_features/sentiment_negative/strided_slice/stack:output:0@dense_features/sentiment_negative/strided_slice/stack_1:output:0@dense_features/sentiment_negative/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features/sentiment_negative/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ű
/dense_features/sentiment_negative/Reshape/shapePack8dense_features/sentiment_negative/strided_slice:output:0:dense_features/sentiment_negative/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:×
)dense_features/sentiment_negative/ReshapeReshape5dense_features/sentiment_negative/ExpandDims:output:08dense_features/sentiment_negative/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
0dense_features/sentiment_positive/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Â
,dense_features/sentiment_positive/ExpandDims
ExpandDimsinputs_sentiment_positive9dense_features/sentiment_positive/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'dense_features/sentiment_positive/ShapeShape5dense_features/sentiment_positive/ExpandDims:output:0*
T0*
_output_shapes
:
5dense_features/sentiment_positive/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7dense_features/sentiment_positive/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7dense_features/sentiment_positive/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ű
/dense_features/sentiment_positive/strided_sliceStridedSlice0dense_features/sentiment_positive/Shape:output:0>dense_features/sentiment_positive/strided_slice/stack:output:0@dense_features/sentiment_positive/strided_slice/stack_1:output:0@dense_features/sentiment_positive/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features/sentiment_positive/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ű
/dense_features/sentiment_positive/Reshape/shapePack8dense_features/sentiment_positive/strided_slice:output:0:dense_features/sentiment_positive/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:×
)dense_features/sentiment_positive/ReshapeReshape5dense_features/sentiment_positive/ExpandDims:output:08dense_features/sentiment_positive/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙|
1dense_features/stopwords_frequency/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ĺ
-dense_features/stopwords_frequency/ExpandDims
ExpandDimsinputs_stopwords_frequency:dense_features/stopwords_frequency/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(dense_features/stopwords_frequency/ShapeShape6dense_features/stopwords_frequency/ExpandDims:output:0*
T0*
_output_shapes
:
6dense_features/stopwords_frequency/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8dense_features/stopwords_frequency/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8dense_features/stopwords_frequency/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0dense_features/stopwords_frequency/strided_sliceStridedSlice1dense_features/stopwords_frequency/Shape:output:0?dense_features/stopwords_frequency/strided_slice/stack:output:0Adense_features/stopwords_frequency/strided_slice/stack_1:output:0Adense_features/stopwords_frequency/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2dense_features/stopwords_frequency/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ţ
0dense_features/stopwords_frequency/Reshape/shapePack9dense_features/stopwords_frequency/strided_slice:output:0;dense_features/stopwords_frequency/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ú
*dense_features/stopwords_frequency/ReshapeReshape6dense_features/stopwords_frequency/ExpandDims:output:09dense_features/stopwords_frequency/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙v
+dense_features/text_standard/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ł
'dense_features/text_standard/ExpandDims
ExpandDimsinputs_text_standard4dense_features/text_standard/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"dense_features/text_standard/ShapeShape0dense_features/text_standard/ExpandDims:output:0*
T0*
_output_shapes
:z
0dense_features/text_standard/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_features/text_standard/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_features/text_standard/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_features/text_standard/strided_sliceStridedSlice+dense_features/text_standard/Shape:output:09dense_features/text_standard/strided_slice/stack:output:0;dense_features/text_standard/strided_slice/stack_1:output:0;dense_features/text_standard/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,dense_features/text_standard/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ě
*dense_features/text_standard/Reshape/shapePack3dense_features/text_standard/strided_slice:output:05dense_features/text_standard/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Č
$dense_features/text_standard/ReshapeReshape0dense_features/text_standard/ExpandDims:output:03dense_features/text_standard/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!dense_features/ttr/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
dense_features/ttr/ExpandDims
ExpandDims
inputs_ttr*dense_features/ttr/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
dense_features/ttr/ShapeShape&dense_features/ttr/ExpandDims:output:0*
T0*
_output_shapes
:p
&dense_features/ttr/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/ttr/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/ttr/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 dense_features/ttr/strided_sliceStridedSlice!dense_features/ttr/Shape:output:0/dense_features/ttr/strided_slice/stack:output:01dense_features/ttr/strided_slice/stack_1:output:01dense_features/ttr/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/ttr/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 dense_features/ttr/Reshape/shapePack)dense_features/ttr/strided_slice:output:0+dense_features/ttr/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
dense_features/ttr/ReshapeReshape&dense_features/ttr/ExpandDims:output:0)dense_features/ttr/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙t
)dense_features/verb_to_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙­
%dense_features/verb_to_adv/ExpandDims
ExpandDimsinputs_verb_to_adv2dense_features/verb_to_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
 dense_features/verb_to_adv/ShapeShape.dense_features/verb_to_adv/ExpandDims:output:0*
T0*
_output_shapes
:x
.dense_features/verb_to_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0dense_features/verb_to_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0dense_features/verb_to_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(dense_features/verb_to_adv/strided_sliceStridedSlice)dense_features/verb_to_adv/Shape:output:07dense_features/verb_to_adv/strided_slice/stack:output:09dense_features/verb_to_adv/strided_slice/stack_1:output:09dense_features/verb_to_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*dense_features/verb_to_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ć
(dense_features/verb_to_adv/Reshape/shapePack1dense_features/verb_to_adv/strided_slice:output:03dense_features/verb_to_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Â
"dense_features/verb_to_adv/ReshapeReshape.dense_features/verb_to_adv/ExpandDims:output:01dense_features/verb_to_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙×
dense_features/concatConcatV2#dense_features/ARI/Reshape:output:04dense_features/Incorrect_form_ratio/Reshape:output:0/dense_features/av_word_per_sen/Reshape:output:0/dense_features/coherence_score/Reshape:output:0<dense_features/dale_chall_readability_score/Reshape:output:04dense_features/flesch_kincaid_grade/Reshape:output:03dense_features/flesch_reading_ease/Reshape:output:0/dense_features/freq_diff_words/Reshape:output:0+dense_features/freq_of_adj/Reshape:output:0+dense_features/freq_of_adv/Reshape:output:04dense_features/freq_of_distinct_adj/Reshape:output:04dense_features/freq_of_distinct_adv/Reshape:output:0,dense_features/freq_of_noun/Reshape:output:0/dense_features/freq_of_pronoun/Reshape:output:02dense_features/freq_of_transition/Reshape:output:0,dense_features/freq_of_verb/Reshape:output:03dense_features/freq_of_wrong_words/Reshape:output:04dense_features/lexrank_avg_min_diff/Reshape:output:05dense_features/lexrank_interquartile/Reshape:output:0.dense_features/mcalpine_eflaw/Reshape:output:0+dense_features/noun_to_adj/Reshape:output:05dense_features/num_of_grammar_errors/Reshape:output:02dense_features/num_of_short_forms/Reshape:output:04dense_features/number_of_diff_words/Reshape:output:0/dense_features/number_of_words/Reshape:output:00dense_features/phrase_diversity/Reshape:output:0,dense_features/punctuations/Reshape:output:03dense_features/sentence_complexity/Reshape:output:02dense_features/sentiment_compound/Reshape:output:02dense_features/sentiment_negative/Reshape:output:02dense_features/sentiment_positive/Reshape:output:03dense_features/stopwords_frequency/Reshape:output:0-dense_features/text_standard/Reshape:output:0#dense_features/ttr/Reshape:output:0+dense_features/verb_to_adv/Reshape:output:0#dense_features/concat/axis:output:0*
N#*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙#
Hidden0/MatMul/ReadVariableOpReadVariableOp&hidden0_matmul_readvariableop_resource*
_output_shapes

:#*
dtype0
Hidden0/MatMulMatMuldense_features/concat:output:0%Hidden0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Hidden0/BiasAdd/ReadVariableOpReadVariableOp'hidden0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Hidden0/BiasAddBiasAddHidden0/MatMul:product:0&Hidden0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
Hidden0/ReluReluHidden0/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
Hidden1/MatMulMatMulHidden0/Relu:activations:0%Hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Hidden1/BiasAddBiasAddHidden1/MatMul:product:0&Hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
Hidden1/ReluReluHidden1/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
Output/MatMulMatMulHidden1/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
IdentityIdentityOutput/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^Hidden0/BiasAdd/ReadVariableOp^Hidden0/MatMul/ReadVariableOp^Hidden1/BiasAdd/ReadVariableOp^Hidden1/MatMul/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ě
_input_shapesş
ˇ:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : 2@
Hidden0/BiasAdd/ReadVariableOpHidden0/BiasAdd/ReadVariableOp2>
Hidden0/MatMul/ReadVariableOpHidden0/MatMul/ReadVariableOp2@
Hidden1/BiasAdd/ReadVariableOpHidden1/BiasAdd/ReadVariableOp2>
Hidden1/MatMul/ReadVariableOpHidden1/MatMul/ReadVariableOp2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp:O K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs/ARI:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/Incorrect_form_ratio:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/av_word_per_sen:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/coherence_score:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs/cohesion:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs/corrected_text:hd
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
=
_user_specified_name%#inputs/dale_chall_readability_score:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/flesch_kincaid_grade:_[
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/flesch_reading_ease:[	W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/freq_diff_words:W
S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/freq_of_adj:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/freq_of_adv:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/freq_of_distinct_adj:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/freq_of_distinct_adv:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs/freq_of_noun:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/freq_of_pronoun:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/freq_of_transition:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs/freq_of_verb:_[
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/freq_of_wrong_words:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/lexrank_avg_min_diff:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_nameinputs/lexrank_interquartile:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs/mcalpine_eflaw:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/noun_to_adj:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_nameinputs/num_of_grammar_errors:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/num_of_short_forms:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/number_of_diff_words:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/number_of_words:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs/phrase_diversity:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs/punctuations:_[
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/sentence_complexity:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/sentiment_compound:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/sentiment_negative:^ Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/sentiment_positive:_![
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/stopwords_frequency:Y"U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs/text_standard:O#K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs/ttr:W$S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/verb_to_adv


ó
B__inference_Hidden0_layer_call_and_return_conditional_losses_51499

inputs0
matmul_readvariableop_resource:#-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:#*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙#
 
_user_specified_nameinputs


ó
B__inference_Hidden1_layer_call_and_return_conditional_losses_54308

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


ó
B__inference_Hidden0_layer_call_and_return_conditional_losses_54288

inputs0
matmul_readvariableop_resource:#-
biasadd_readvariableop_resource:
identity˘BiasAdd/ReadVariableOp˘MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:#*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:˙˙˙˙˙˙˙˙˙#: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:˙˙˙˙˙˙˙˙˙#
 
_user_specified_nameinputs
Ć<
	
G__inference_sequential_2_layer_call_and_return_conditional_losses_52335
ari	
incorrect_form_ratio
av_word_per_sen
coherence_score
cohesion
corrected_text 
dale_chall_readability_score
flesch_kincaid_grade
flesch_reading_ease
freq_diff_words
freq_of_adj
freq_of_adv
freq_of_distinct_adj
freq_of_distinct_adv
freq_of_noun
freq_of_pronoun
freq_of_transition
freq_of_verb
freq_of_wrong_words
lexrank_avg_min_diff
lexrank_interquartile
mcalpine_eflaw
noun_to_adj
num_of_grammar_errors	
num_of_short_forms	
number_of_diff_words	
number_of_words	
phrase_diversity
punctuations
sentence_complexity
sentiment_compound
sentiment_negative
sentiment_positive
stopwords_frequency
text_standard
ttr
verb_to_adv
hidden0_52319:#
hidden0_52321:
hidden1_52324:
hidden1_52326:
output_52329:
output_52331:
identity˘Hidden0/StatefulPartitionedCall˘Hidden1/StatefulPartitionedCall˘Output/StatefulPartitionedCallď
dense_features/PartitionedCallPartitionedCallariincorrect_form_ratioav_word_per_sencoherence_scorecohesioncorrected_textdale_chall_readability_scoreflesch_kincaid_gradeflesch_reading_easefreq_diff_wordsfreq_of_adjfreq_of_advfreq_of_distinct_adjfreq_of_distinct_advfreq_of_nounfreq_of_pronounfreq_of_transitionfreq_of_verbfreq_of_wrong_wordslexrank_avg_min_difflexrank_interquartilemcalpine_eflawnoun_to_adjnum_of_grammar_errorsnum_of_short_formsnumber_of_diff_wordsnumber_of_wordsphrase_diversitypunctuationssentence_complexitysentiment_compoundsentiment_negativesentiment_positivestopwords_frequencytext_standardttrverb_to_adv*0
Tin)
'2%					*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙#* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_dense_features_layer_call_and_return_conditional_losses_51486
Hidden0/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0hidden0_52319hidden0_52321*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Hidden0_layer_call_and_return_conditional_losses_51499
Hidden1/StatefulPartitionedCallStatefulPartitionedCall(Hidden0/StatefulPartitionedCall:output:0hidden1_52324hidden1_52326*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_Hidden1_layer_call_and_return_conditional_losses_51516
Output/StatefulPartitionedCallStatefulPartitionedCall(Hidden1/StatefulPartitionedCall:output:0output_52329output_52331*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_Output_layer_call_and_return_conditional_losses_51532v
IdentityIdentity'Output/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ť
NoOpNoOp ^Hidden0/StatefulPartitionedCall ^Hidden1/StatefulPartitionedCall^Output/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ě
_input_shapesş
ˇ:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : 2B
Hidden0/StatefulPartitionedCallHidden0/StatefulPartitionedCall2B
Hidden1/StatefulPartitionedCallHidden1/StatefulPartitionedCall2@
Output/StatefulPartitionedCallOutput/StatefulPartitionedCall:H D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_nameARI:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameIncorrect_form_ratio:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameav_word_per_sen:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namecoherence_score:MI
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
"
_user_specified_name
cohesion:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namecorrected_text:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_namedale_chall_readability_score:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameflesch_kincaid_grade:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameflesch_reading_ease:T	P
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefreq_diff_words:P
L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefreq_of_adj:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namefreq_of_adv:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefreq_of_distinct_adj:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namefreq_of_distinct_adv:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefreq_of_noun:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namefreq_of_pronoun:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namefreq_of_transition:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namefreq_of_verb:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namefreq_of_wrong_words:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namelexrank_avg_min_diff:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namelexrank_interquartile:SO
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
(
_user_specified_namemcalpine_eflaw:PL
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_namenoun_to_adj:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_namenum_of_grammar_errors:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namenum_of_short_forms:YU
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_namenumber_of_diff_words:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_namenumber_of_words:UQ
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
_user_specified_namephrase_diversity:QM
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
&
_user_specified_namepunctuations:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namesentence_complexity:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_compound:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_negative:W S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_namesentiment_positive:X!T
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_namestopwords_frequency:R"N
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
'
_user_specified_nametext_standard:H#D
#
_output_shapes
:˙˙˙˙˙˙˙˙˙

_user_specified_namettr:P$L
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
%
_user_specified_nameverb_to_adv
Đś

G__inference_sequential_2_layer_call_and_return_conditional_losses_53392

inputs_ari	
inputs_incorrect_form_ratio
inputs_av_word_per_sen
inputs_coherence_score
inputs_cohesion
inputs_corrected_text'
#inputs_dale_chall_readability_score
inputs_flesch_kincaid_grade
inputs_flesch_reading_ease
inputs_freq_diff_words
inputs_freq_of_adj
inputs_freq_of_adv
inputs_freq_of_distinct_adj
inputs_freq_of_distinct_adv
inputs_freq_of_noun
inputs_freq_of_pronoun
inputs_freq_of_transition
inputs_freq_of_verb
inputs_freq_of_wrong_words
inputs_lexrank_avg_min_diff 
inputs_lexrank_interquartile
inputs_mcalpine_eflaw
inputs_noun_to_adj 
inputs_num_of_grammar_errors	
inputs_num_of_short_forms	
inputs_number_of_diff_words	
inputs_number_of_words	
inputs_phrase_diversity
inputs_punctuations
inputs_sentence_complexity
inputs_sentiment_compound
inputs_sentiment_negative
inputs_sentiment_positive
inputs_stopwords_frequency
inputs_text_standard

inputs_ttr
inputs_verb_to_adv8
&hidden0_matmul_readvariableop_resource:#5
'hidden0_biasadd_readvariableop_resource:8
&hidden1_matmul_readvariableop_resource:5
'hidden1_biasadd_readvariableop_resource:7
%output_matmul_readvariableop_resource:4
&output_biasadd_readvariableop_resource:
identity˘Hidden0/BiasAdd/ReadVariableOp˘Hidden0/MatMul/ReadVariableOp˘Hidden1/BiasAdd/ReadVariableOp˘Hidden1/MatMul/ReadVariableOp˘Output/BiasAdd/ReadVariableOp˘Output/MatMul/ReadVariableOpl
!dense_features/ARI/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
dense_features/ARI/ExpandDims
ExpandDims
inputs_ari*dense_features/ARI/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
dense_features/ARI/CastCast&dense_features/ARI/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙c
dense_features/ARI/ShapeShapedense_features/ARI/Cast:y:0*
T0*
_output_shapes
:p
&dense_features/ARI/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/ARI/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/ARI/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 dense_features/ARI/strided_sliceStridedSlice!dense_features/ARI/Shape:output:0/dense_features/ARI/strided_slice/stack:output:01dense_features/ARI/strided_slice/stack_1:output:01dense_features/ARI/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/ARI/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 dense_features/ARI/Reshape/shapePack)dense_features/ARI/strided_slice:output:0+dense_features/ARI/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:
dense_features/ARI/ReshapeReshapedense_features/ARI/Cast:y:0)dense_features/ARI/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙}
2dense_features/Incorrect_form_ratio/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Č
.dense_features/Incorrect_form_ratio/ExpandDims
ExpandDimsinputs_incorrect_form_ratio;dense_features/Incorrect_form_ratio/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)dense_features/Incorrect_form_ratio/ShapeShape7dense_features/Incorrect_form_ratio/ExpandDims:output:0*
T0*
_output_shapes
:
7dense_features/Incorrect_form_ratio/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9dense_features/Incorrect_form_ratio/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9dense_features/Incorrect_form_ratio/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1dense_features/Incorrect_form_ratio/strided_sliceStridedSlice2dense_features/Incorrect_form_ratio/Shape:output:0@dense_features/Incorrect_form_ratio/strided_slice/stack:output:0Bdense_features/Incorrect_form_ratio/strided_slice/stack_1:output:0Bdense_features/Incorrect_form_ratio/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/Incorrect_form_ratio/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :á
1dense_features/Incorrect_form_ratio/Reshape/shapePack:dense_features/Incorrect_form_ratio/strided_slice:output:0<dense_features/Incorrect_form_ratio/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ý
+dense_features/Incorrect_form_ratio/ReshapeReshape7dense_features/Incorrect_form_ratio/ExpandDims:output:0:dense_features/Incorrect_form_ratio/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙x
-dense_features/av_word_per_sen/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙š
)dense_features/av_word_per_sen/ExpandDims
ExpandDimsinputs_av_word_per_sen6dense_features/av_word_per_sen/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$dense_features/av_word_per_sen/ShapeShape2dense_features/av_word_per_sen/ExpandDims:output:0*
T0*
_output_shapes
:|
2dense_features/av_word_per_sen/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/av_word_per_sen/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/av_word_per_sen/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
,dense_features/av_word_per_sen/strided_sliceStridedSlice-dense_features/av_word_per_sen/Shape:output:0;dense_features/av_word_per_sen/strided_slice/stack:output:0=dense_features/av_word_per_sen/strided_slice/stack_1:output:0=dense_features/av_word_per_sen/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/av_word_per_sen/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ň
,dense_features/av_word_per_sen/Reshape/shapePack5dense_features/av_word_per_sen/strided_slice:output:07dense_features/av_word_per_sen/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Î
&dense_features/av_word_per_sen/ReshapeReshape2dense_features/av_word_per_sen/ExpandDims:output:05dense_features/av_word_per_sen/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙x
-dense_features/coherence_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙š
)dense_features/coherence_score/ExpandDims
ExpandDimsinputs_coherence_score6dense_features/coherence_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$dense_features/coherence_score/ShapeShape2dense_features/coherence_score/ExpandDims:output:0*
T0*
_output_shapes
:|
2dense_features/coherence_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/coherence_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/coherence_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
,dense_features/coherence_score/strided_sliceStridedSlice-dense_features/coherence_score/Shape:output:0;dense_features/coherence_score/strided_slice/stack:output:0=dense_features/coherence_score/strided_slice/stack_1:output:0=dense_features/coherence_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/coherence_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ň
,dense_features/coherence_score/Reshape/shapePack5dense_features/coherence_score/strided_slice:output:07dense_features/coherence_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Î
&dense_features/coherence_score/ReshapeReshape2dense_features/coherence_score/ExpandDims:output:05dense_features/coherence_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
:dense_features/dale_chall_readability_score/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ŕ
6dense_features/dale_chall_readability_score/ExpandDims
ExpandDims#inputs_dale_chall_readability_scoreCdense_features/dale_chall_readability_score/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
1dense_features/dale_chall_readability_score/ShapeShape?dense_features/dale_chall_readability_score/ExpandDims:output:0*
T0*
_output_shapes
:
?dense_features/dale_chall_readability_score/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Adense_features/dale_chall_readability_score/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Adense_features/dale_chall_readability_score/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
9dense_features/dale_chall_readability_score/strided_sliceStridedSlice:dense_features/dale_chall_readability_score/Shape:output:0Hdense_features/dale_chall_readability_score/strided_slice/stack:output:0Jdense_features/dale_chall_readability_score/strided_slice/stack_1:output:0Jdense_features/dale_chall_readability_score/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask}
;dense_features/dale_chall_readability_score/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ů
9dense_features/dale_chall_readability_score/Reshape/shapePackBdense_features/dale_chall_readability_score/strided_slice:output:0Ddense_features/dale_chall_readability_score/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ő
3dense_features/dale_chall_readability_score/ReshapeReshape?dense_features/dale_chall_readability_score/ExpandDims:output:0Bdense_features/dale_chall_readability_score/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙}
2dense_features/flesch_kincaid_grade/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Č
.dense_features/flesch_kincaid_grade/ExpandDims
ExpandDimsinputs_flesch_kincaid_grade;dense_features/flesch_kincaid_grade/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)dense_features/flesch_kincaid_grade/ShapeShape7dense_features/flesch_kincaid_grade/ExpandDims:output:0*
T0*
_output_shapes
:
7dense_features/flesch_kincaid_grade/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9dense_features/flesch_kincaid_grade/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9dense_features/flesch_kincaid_grade/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1dense_features/flesch_kincaid_grade/strided_sliceStridedSlice2dense_features/flesch_kincaid_grade/Shape:output:0@dense_features/flesch_kincaid_grade/strided_slice/stack:output:0Bdense_features/flesch_kincaid_grade/strided_slice/stack_1:output:0Bdense_features/flesch_kincaid_grade/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/flesch_kincaid_grade/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :á
1dense_features/flesch_kincaid_grade/Reshape/shapePack:dense_features/flesch_kincaid_grade/strided_slice:output:0<dense_features/flesch_kincaid_grade/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ý
+dense_features/flesch_kincaid_grade/ReshapeReshape7dense_features/flesch_kincaid_grade/ExpandDims:output:0:dense_features/flesch_kincaid_grade/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙|
1dense_features/flesch_reading_ease/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ĺ
-dense_features/flesch_reading_ease/ExpandDims
ExpandDimsinputs_flesch_reading_ease:dense_features/flesch_reading_ease/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(dense_features/flesch_reading_ease/ShapeShape6dense_features/flesch_reading_ease/ExpandDims:output:0*
T0*
_output_shapes
:
6dense_features/flesch_reading_ease/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8dense_features/flesch_reading_ease/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8dense_features/flesch_reading_ease/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0dense_features/flesch_reading_ease/strided_sliceStridedSlice1dense_features/flesch_reading_ease/Shape:output:0?dense_features/flesch_reading_ease/strided_slice/stack:output:0Adense_features/flesch_reading_ease/strided_slice/stack_1:output:0Adense_features/flesch_reading_ease/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2dense_features/flesch_reading_ease/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ţ
0dense_features/flesch_reading_ease/Reshape/shapePack9dense_features/flesch_reading_ease/strided_slice:output:0;dense_features/flesch_reading_ease/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ú
*dense_features/flesch_reading_ease/ReshapeReshape6dense_features/flesch_reading_ease/ExpandDims:output:09dense_features/flesch_reading_ease/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙x
-dense_features/freq_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙š
)dense_features/freq_diff_words/ExpandDims
ExpandDimsinputs_freq_diff_words6dense_features/freq_diff_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$dense_features/freq_diff_words/ShapeShape2dense_features/freq_diff_words/ExpandDims:output:0*
T0*
_output_shapes
:|
2dense_features/freq_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/freq_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/freq_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
,dense_features/freq_diff_words/strided_sliceStridedSlice-dense_features/freq_diff_words/Shape:output:0;dense_features/freq_diff_words/strided_slice/stack:output:0=dense_features/freq_diff_words/strided_slice/stack_1:output:0=dense_features/freq_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/freq_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ň
,dense_features/freq_diff_words/Reshape/shapePack5dense_features/freq_diff_words/strided_slice:output:07dense_features/freq_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Î
&dense_features/freq_diff_words/ReshapeReshape2dense_features/freq_diff_words/ExpandDims:output:05dense_features/freq_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙t
)dense_features/freq_of_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙­
%dense_features/freq_of_adj/ExpandDims
ExpandDimsinputs_freq_of_adj2dense_features/freq_of_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
 dense_features/freq_of_adj/ShapeShape.dense_features/freq_of_adj/ExpandDims:output:0*
T0*
_output_shapes
:x
.dense_features/freq_of_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0dense_features/freq_of_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0dense_features/freq_of_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(dense_features/freq_of_adj/strided_sliceStridedSlice)dense_features/freq_of_adj/Shape:output:07dense_features/freq_of_adj/strided_slice/stack:output:09dense_features/freq_of_adj/strided_slice/stack_1:output:09dense_features/freq_of_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*dense_features/freq_of_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ć
(dense_features/freq_of_adj/Reshape/shapePack1dense_features/freq_of_adj/strided_slice:output:03dense_features/freq_of_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Â
"dense_features/freq_of_adj/ReshapeReshape.dense_features/freq_of_adj/ExpandDims:output:01dense_features/freq_of_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙t
)dense_features/freq_of_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙­
%dense_features/freq_of_adv/ExpandDims
ExpandDimsinputs_freq_of_adv2dense_features/freq_of_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
 dense_features/freq_of_adv/ShapeShape.dense_features/freq_of_adv/ExpandDims:output:0*
T0*
_output_shapes
:x
.dense_features/freq_of_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0dense_features/freq_of_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0dense_features/freq_of_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(dense_features/freq_of_adv/strided_sliceStridedSlice)dense_features/freq_of_adv/Shape:output:07dense_features/freq_of_adv/strided_slice/stack:output:09dense_features/freq_of_adv/strided_slice/stack_1:output:09dense_features/freq_of_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*dense_features/freq_of_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ć
(dense_features/freq_of_adv/Reshape/shapePack1dense_features/freq_of_adv/strided_slice:output:03dense_features/freq_of_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Â
"dense_features/freq_of_adv/ReshapeReshape.dense_features/freq_of_adv/ExpandDims:output:01dense_features/freq_of_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙}
2dense_features/freq_of_distinct_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Č
.dense_features/freq_of_distinct_adj/ExpandDims
ExpandDimsinputs_freq_of_distinct_adj;dense_features/freq_of_distinct_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)dense_features/freq_of_distinct_adj/ShapeShape7dense_features/freq_of_distinct_adj/ExpandDims:output:0*
T0*
_output_shapes
:
7dense_features/freq_of_distinct_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9dense_features/freq_of_distinct_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9dense_features/freq_of_distinct_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1dense_features/freq_of_distinct_adj/strided_sliceStridedSlice2dense_features/freq_of_distinct_adj/Shape:output:0@dense_features/freq_of_distinct_adj/strided_slice/stack:output:0Bdense_features/freq_of_distinct_adj/strided_slice/stack_1:output:0Bdense_features/freq_of_distinct_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/freq_of_distinct_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :á
1dense_features/freq_of_distinct_adj/Reshape/shapePack:dense_features/freq_of_distinct_adj/strided_slice:output:0<dense_features/freq_of_distinct_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ý
+dense_features/freq_of_distinct_adj/ReshapeReshape7dense_features/freq_of_distinct_adj/ExpandDims:output:0:dense_features/freq_of_distinct_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙}
2dense_features/freq_of_distinct_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Č
.dense_features/freq_of_distinct_adv/ExpandDims
ExpandDimsinputs_freq_of_distinct_adv;dense_features/freq_of_distinct_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)dense_features/freq_of_distinct_adv/ShapeShape7dense_features/freq_of_distinct_adv/ExpandDims:output:0*
T0*
_output_shapes
:
7dense_features/freq_of_distinct_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9dense_features/freq_of_distinct_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9dense_features/freq_of_distinct_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1dense_features/freq_of_distinct_adv/strided_sliceStridedSlice2dense_features/freq_of_distinct_adv/Shape:output:0@dense_features/freq_of_distinct_adv/strided_slice/stack:output:0Bdense_features/freq_of_distinct_adv/strided_slice/stack_1:output:0Bdense_features/freq_of_distinct_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/freq_of_distinct_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :á
1dense_features/freq_of_distinct_adv/Reshape/shapePack:dense_features/freq_of_distinct_adv/strided_slice:output:0<dense_features/freq_of_distinct_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ý
+dense_features/freq_of_distinct_adv/ReshapeReshape7dense_features/freq_of_distinct_adv/ExpandDims:output:0:dense_features/freq_of_distinct_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙u
*dense_features/freq_of_noun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙°
&dense_features/freq_of_noun/ExpandDims
ExpandDimsinputs_freq_of_noun3dense_features/freq_of_noun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!dense_features/freq_of_noun/ShapeShape/dense_features/freq_of_noun/ExpandDims:output:0*
T0*
_output_shapes
:y
/dense_features/freq_of_noun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/freq_of_noun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/freq_of_noun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)dense_features/freq_of_noun/strided_sliceStridedSlice*dense_features/freq_of_noun/Shape:output:08dense_features/freq_of_noun/strided_slice/stack:output:0:dense_features/freq_of_noun/strided_slice/stack_1:output:0:dense_features/freq_of_noun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/freq_of_noun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :É
)dense_features/freq_of_noun/Reshape/shapePack2dense_features/freq_of_noun/strided_slice:output:04dense_features/freq_of_noun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ĺ
#dense_features/freq_of_noun/ReshapeReshape/dense_features/freq_of_noun/ExpandDims:output:02dense_features/freq_of_noun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙x
-dense_features/freq_of_pronoun/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙š
)dense_features/freq_of_pronoun/ExpandDims
ExpandDimsinputs_freq_of_pronoun6dense_features/freq_of_pronoun/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
$dense_features/freq_of_pronoun/ShapeShape2dense_features/freq_of_pronoun/ExpandDims:output:0*
T0*
_output_shapes
:|
2dense_features/freq_of_pronoun/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/freq_of_pronoun/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/freq_of_pronoun/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
,dense_features/freq_of_pronoun/strided_sliceStridedSlice-dense_features/freq_of_pronoun/Shape:output:0;dense_features/freq_of_pronoun/strided_slice/stack:output:0=dense_features/freq_of_pronoun/strided_slice/stack_1:output:0=dense_features/freq_of_pronoun/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/freq_of_pronoun/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ň
,dense_features/freq_of_pronoun/Reshape/shapePack5dense_features/freq_of_pronoun/strided_slice:output:07dense_features/freq_of_pronoun/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Î
&dense_features/freq_of_pronoun/ReshapeReshape2dense_features/freq_of_pronoun/ExpandDims:output:05dense_features/freq_of_pronoun/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
0dense_features/freq_of_transition/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Â
,dense_features/freq_of_transition/ExpandDims
ExpandDimsinputs_freq_of_transition9dense_features/freq_of_transition/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'dense_features/freq_of_transition/ShapeShape5dense_features/freq_of_transition/ExpandDims:output:0*
T0*
_output_shapes
:
5dense_features/freq_of_transition/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7dense_features/freq_of_transition/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7dense_features/freq_of_transition/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ű
/dense_features/freq_of_transition/strided_sliceStridedSlice0dense_features/freq_of_transition/Shape:output:0>dense_features/freq_of_transition/strided_slice/stack:output:0@dense_features/freq_of_transition/strided_slice/stack_1:output:0@dense_features/freq_of_transition/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features/freq_of_transition/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ű
/dense_features/freq_of_transition/Reshape/shapePack8dense_features/freq_of_transition/strided_slice:output:0:dense_features/freq_of_transition/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:×
)dense_features/freq_of_transition/ReshapeReshape5dense_features/freq_of_transition/ExpandDims:output:08dense_features/freq_of_transition/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙u
*dense_features/freq_of_verb/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙°
&dense_features/freq_of_verb/ExpandDims
ExpandDimsinputs_freq_of_verb3dense_features/freq_of_verb/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!dense_features/freq_of_verb/ShapeShape/dense_features/freq_of_verb/ExpandDims:output:0*
T0*
_output_shapes
:y
/dense_features/freq_of_verb/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/freq_of_verb/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/freq_of_verb/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)dense_features/freq_of_verb/strided_sliceStridedSlice*dense_features/freq_of_verb/Shape:output:08dense_features/freq_of_verb/strided_slice/stack:output:0:dense_features/freq_of_verb/strided_slice/stack_1:output:0:dense_features/freq_of_verb/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/freq_of_verb/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :É
)dense_features/freq_of_verb/Reshape/shapePack2dense_features/freq_of_verb/strided_slice:output:04dense_features/freq_of_verb/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ĺ
#dense_features/freq_of_verb/ReshapeReshape/dense_features/freq_of_verb/ExpandDims:output:02dense_features/freq_of_verb/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙|
1dense_features/freq_of_wrong_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ĺ
-dense_features/freq_of_wrong_words/ExpandDims
ExpandDimsinputs_freq_of_wrong_words:dense_features/freq_of_wrong_words/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(dense_features/freq_of_wrong_words/ShapeShape6dense_features/freq_of_wrong_words/ExpandDims:output:0*
T0*
_output_shapes
:
6dense_features/freq_of_wrong_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8dense_features/freq_of_wrong_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8dense_features/freq_of_wrong_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0dense_features/freq_of_wrong_words/strided_sliceStridedSlice1dense_features/freq_of_wrong_words/Shape:output:0?dense_features/freq_of_wrong_words/strided_slice/stack:output:0Adense_features/freq_of_wrong_words/strided_slice/stack_1:output:0Adense_features/freq_of_wrong_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2dense_features/freq_of_wrong_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ţ
0dense_features/freq_of_wrong_words/Reshape/shapePack9dense_features/freq_of_wrong_words/strided_slice:output:0;dense_features/freq_of_wrong_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ú
*dense_features/freq_of_wrong_words/ReshapeReshape6dense_features/freq_of_wrong_words/ExpandDims:output:09dense_features/freq_of_wrong_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙}
2dense_features/lexrank_avg_min_diff/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Č
.dense_features/lexrank_avg_min_diff/ExpandDims
ExpandDimsinputs_lexrank_avg_min_diff;dense_features/lexrank_avg_min_diff/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)dense_features/lexrank_avg_min_diff/ShapeShape7dense_features/lexrank_avg_min_diff/ExpandDims:output:0*
T0*
_output_shapes
:
7dense_features/lexrank_avg_min_diff/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9dense_features/lexrank_avg_min_diff/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9dense_features/lexrank_avg_min_diff/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1dense_features/lexrank_avg_min_diff/strided_sliceStridedSlice2dense_features/lexrank_avg_min_diff/Shape:output:0@dense_features/lexrank_avg_min_diff/strided_slice/stack:output:0Bdense_features/lexrank_avg_min_diff/strided_slice/stack_1:output:0Bdense_features/lexrank_avg_min_diff/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/lexrank_avg_min_diff/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :á
1dense_features/lexrank_avg_min_diff/Reshape/shapePack:dense_features/lexrank_avg_min_diff/strided_slice:output:0<dense_features/lexrank_avg_min_diff/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ý
+dense_features/lexrank_avg_min_diff/ReshapeReshape7dense_features/lexrank_avg_min_diff/ExpandDims:output:0:dense_features/lexrank_avg_min_diff/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
3dense_features/lexrank_interquartile/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ë
/dense_features/lexrank_interquartile/ExpandDims
ExpandDimsinputs_lexrank_interquartile<dense_features/lexrank_interquartile/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*dense_features/lexrank_interquartile/ShapeShape8dense_features/lexrank_interquartile/ExpandDims:output:0*
T0*
_output_shapes
:
8dense_features/lexrank_interquartile/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:dense_features/lexrank_interquartile/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:dense_features/lexrank_interquartile/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2dense_features/lexrank_interquartile/strided_sliceStridedSlice3dense_features/lexrank_interquartile/Shape:output:0Adense_features/lexrank_interquartile/strided_slice/stack:output:0Cdense_features/lexrank_interquartile/strided_slice/stack_1:output:0Cdense_features/lexrank_interquartile/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/lexrank_interquartile/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ä
2dense_features/lexrank_interquartile/Reshape/shapePack;dense_features/lexrank_interquartile/strided_slice:output:0=dense_features/lexrank_interquartile/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ŕ
,dense_features/lexrank_interquartile/ReshapeReshape8dense_features/lexrank_interquartile/ExpandDims:output:0;dense_features/lexrank_interquartile/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙w
,dense_features/mcalpine_eflaw/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ś
(dense_features/mcalpine_eflaw/ExpandDims
ExpandDimsinputs_mcalpine_eflaw5dense_features/mcalpine_eflaw/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
#dense_features/mcalpine_eflaw/ShapeShape1dense_features/mcalpine_eflaw/ExpandDims:output:0*
T0*
_output_shapes
:{
1dense_features/mcalpine_eflaw/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_features/mcalpine_eflaw/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_features/mcalpine_eflaw/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+dense_features/mcalpine_eflaw/strided_sliceStridedSlice,dense_features/mcalpine_eflaw/Shape:output:0:dense_features/mcalpine_eflaw/strided_slice/stack:output:0<dense_features/mcalpine_eflaw/strided_slice/stack_1:output:0<dense_features/mcalpine_eflaw/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-dense_features/mcalpine_eflaw/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ď
+dense_features/mcalpine_eflaw/Reshape/shapePack4dense_features/mcalpine_eflaw/strided_slice:output:06dense_features/mcalpine_eflaw/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ë
%dense_features/mcalpine_eflaw/ReshapeReshape1dense_features/mcalpine_eflaw/ExpandDims:output:04dense_features/mcalpine_eflaw/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙t
)dense_features/noun_to_adj/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙­
%dense_features/noun_to_adj/ExpandDims
ExpandDimsinputs_noun_to_adj2dense_features/noun_to_adj/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
 dense_features/noun_to_adj/ShapeShape.dense_features/noun_to_adj/ExpandDims:output:0*
T0*
_output_shapes
:x
.dense_features/noun_to_adj/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0dense_features/noun_to_adj/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0dense_features/noun_to_adj/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(dense_features/noun_to_adj/strided_sliceStridedSlice)dense_features/noun_to_adj/Shape:output:07dense_features/noun_to_adj/strided_slice/stack:output:09dense_features/noun_to_adj/strided_slice/stack_1:output:09dense_features/noun_to_adj/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*dense_features/noun_to_adj/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ć
(dense_features/noun_to_adj/Reshape/shapePack1dense_features/noun_to_adj/strided_slice:output:03dense_features/noun_to_adj/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Â
"dense_features/noun_to_adj/ReshapeReshape.dense_features/noun_to_adj/ExpandDims:output:01dense_features/noun_to_adj/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
3dense_features/num_of_grammar_errors/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ë
/dense_features/num_of_grammar_errors/ExpandDims
ExpandDimsinputs_num_of_grammar_errors<dense_features/num_of_grammar_errors/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ź
)dense_features/num_of_grammar_errors/CastCast8dense_features/num_of_grammar_errors/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*dense_features/num_of_grammar_errors/ShapeShape-dense_features/num_of_grammar_errors/Cast:y:0*
T0*
_output_shapes
:
8dense_features/num_of_grammar_errors/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:dense_features/num_of_grammar_errors/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:dense_features/num_of_grammar_errors/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2dense_features/num_of_grammar_errors/strided_sliceStridedSlice3dense_features/num_of_grammar_errors/Shape:output:0Adense_features/num_of_grammar_errors/strided_slice/stack:output:0Cdense_features/num_of_grammar_errors/strided_slice/stack_1:output:0Cdense_features/num_of_grammar_errors/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
4dense_features/num_of_grammar_errors/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ä
2dense_features/num_of_grammar_errors/Reshape/shapePack;dense_features/num_of_grammar_errors/strided_slice:output:0=dense_features/num_of_grammar_errors/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ő
,dense_features/num_of_grammar_errors/ReshapeReshape-dense_features/num_of_grammar_errors/Cast:y:0;dense_features/num_of_grammar_errors/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
0dense_features/num_of_short_forms/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Â
,dense_features/num_of_short_forms/ExpandDims
ExpandDimsinputs_num_of_short_forms9dense_features/num_of_short_forms/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ś
&dense_features/num_of_short_forms/CastCast5dense_features/num_of_short_forms/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'dense_features/num_of_short_forms/ShapeShape*dense_features/num_of_short_forms/Cast:y:0*
T0*
_output_shapes
:
5dense_features/num_of_short_forms/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7dense_features/num_of_short_forms/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7dense_features/num_of_short_forms/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ű
/dense_features/num_of_short_forms/strided_sliceStridedSlice0dense_features/num_of_short_forms/Shape:output:0>dense_features/num_of_short_forms/strided_slice/stack:output:0@dense_features/num_of_short_forms/strided_slice/stack_1:output:0@dense_features/num_of_short_forms/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features/num_of_short_forms/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ű
/dense_features/num_of_short_forms/Reshape/shapePack8dense_features/num_of_short_forms/strided_slice:output:0:dense_features/num_of_short_forms/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ě
)dense_features/num_of_short_forms/ReshapeReshape*dense_features/num_of_short_forms/Cast:y:08dense_features/num_of_short_forms/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙}
2dense_features/number_of_diff_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Č
.dense_features/number_of_diff_words/ExpandDims
ExpandDimsinputs_number_of_diff_words;dense_features/number_of_diff_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙Ş
(dense_features/number_of_diff_words/CastCast7dense_features/number_of_diff_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
)dense_features/number_of_diff_words/ShapeShape,dense_features/number_of_diff_words/Cast:y:0*
T0*
_output_shapes
:
7dense_features/number_of_diff_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9dense_features/number_of_diff_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9dense_features/number_of_diff_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1dense_features/number_of_diff_words/strided_sliceStridedSlice2dense_features/number_of_diff_words/Shape:output:0@dense_features/number_of_diff_words/strided_slice/stack:output:0Bdense_features/number_of_diff_words/strided_slice/stack_1:output:0Bdense_features/number_of_diff_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3dense_features/number_of_diff_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :á
1dense_features/number_of_diff_words/Reshape/shapePack:dense_features/number_of_diff_words/strided_slice:output:0<dense_features/number_of_diff_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ň
+dense_features/number_of_diff_words/ReshapeReshape,dense_features/number_of_diff_words/Cast:y:0:dense_features/number_of_diff_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙x
-dense_features/number_of_words/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙š
)dense_features/number_of_words/ExpandDims
ExpandDimsinputs_number_of_words6dense_features/number_of_words/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
#dense_features/number_of_words/CastCast2dense_features/number_of_words/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
$dense_features/number_of_words/ShapeShape'dense_features/number_of_words/Cast:y:0*
T0*
_output_shapes
:|
2dense_features/number_of_words/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4dense_features/number_of_words/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4dense_features/number_of_words/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ě
,dense_features/number_of_words/strided_sliceStridedSlice-dense_features/number_of_words/Shape:output:0;dense_features/number_of_words/strided_slice/stack:output:0=dense_features/number_of_words/strided_slice/stack_1:output:0=dense_features/number_of_words/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskp
.dense_features/number_of_words/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ň
,dense_features/number_of_words/Reshape/shapePack5dense_features/number_of_words/strided_slice:output:07dense_features/number_of_words/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ă
&dense_features/number_of_words/ReshapeReshape'dense_features/number_of_words/Cast:y:05dense_features/number_of_words/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙y
.dense_features/phrase_diversity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ź
*dense_features/phrase_diversity/ExpandDims
ExpandDimsinputs_phrase_diversity7dense_features/phrase_diversity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
%dense_features/phrase_diversity/ShapeShape3dense_features/phrase_diversity/ExpandDims:output:0*
T0*
_output_shapes
:}
3dense_features/phrase_diversity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_features/phrase_diversity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_features/phrase_diversity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ń
-dense_features/phrase_diversity/strided_sliceStridedSlice.dense_features/phrase_diversity/Shape:output:0<dense_features/phrase_diversity/strided_slice/stack:output:0>dense_features/phrase_diversity/strided_slice/stack_1:output:0>dense_features/phrase_diversity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/dense_features/phrase_diversity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ő
-dense_features/phrase_diversity/Reshape/shapePack6dense_features/phrase_diversity/strided_slice:output:08dense_features/phrase_diversity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ń
'dense_features/phrase_diversity/ReshapeReshape3dense_features/phrase_diversity/ExpandDims:output:06dense_features/phrase_diversity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙u
*dense_features/punctuations/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙°
&dense_features/punctuations/ExpandDims
ExpandDimsinputs_punctuations3dense_features/punctuations/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
!dense_features/punctuations/ShapeShape/dense_features/punctuations/ExpandDims:output:0*
T0*
_output_shapes
:y
/dense_features/punctuations/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/punctuations/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/punctuations/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)dense_features/punctuations/strided_sliceStridedSlice*dense_features/punctuations/Shape:output:08dense_features/punctuations/strided_slice/stack:output:0:dense_features/punctuations/strided_slice/stack_1:output:0:dense_features/punctuations/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/punctuations/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :É
)dense_features/punctuations/Reshape/shapePack2dense_features/punctuations/strided_slice:output:04dense_features/punctuations/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ĺ
#dense_features/punctuations/ReshapeReshape/dense_features/punctuations/ExpandDims:output:02dense_features/punctuations/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙|
1dense_features/sentence_complexity/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ĺ
-dense_features/sentence_complexity/ExpandDims
ExpandDimsinputs_sentence_complexity:dense_features/sentence_complexity/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(dense_features/sentence_complexity/ShapeShape6dense_features/sentence_complexity/ExpandDims:output:0*
T0*
_output_shapes
:
6dense_features/sentence_complexity/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8dense_features/sentence_complexity/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8dense_features/sentence_complexity/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0dense_features/sentence_complexity/strided_sliceStridedSlice1dense_features/sentence_complexity/Shape:output:0?dense_features/sentence_complexity/strided_slice/stack:output:0Adense_features/sentence_complexity/strided_slice/stack_1:output:0Adense_features/sentence_complexity/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2dense_features/sentence_complexity/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ţ
0dense_features/sentence_complexity/Reshape/shapePack9dense_features/sentence_complexity/strided_slice:output:0;dense_features/sentence_complexity/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ú
*dense_features/sentence_complexity/ReshapeReshape6dense_features/sentence_complexity/ExpandDims:output:09dense_features/sentence_complexity/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
0dense_features/sentiment_compound/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Â
,dense_features/sentiment_compound/ExpandDims
ExpandDimsinputs_sentiment_compound9dense_features/sentiment_compound/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'dense_features/sentiment_compound/ShapeShape5dense_features/sentiment_compound/ExpandDims:output:0*
T0*
_output_shapes
:
5dense_features/sentiment_compound/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7dense_features/sentiment_compound/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7dense_features/sentiment_compound/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ű
/dense_features/sentiment_compound/strided_sliceStridedSlice0dense_features/sentiment_compound/Shape:output:0>dense_features/sentiment_compound/strided_slice/stack:output:0@dense_features/sentiment_compound/strided_slice/stack_1:output:0@dense_features/sentiment_compound/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features/sentiment_compound/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ű
/dense_features/sentiment_compound/Reshape/shapePack8dense_features/sentiment_compound/strided_slice:output:0:dense_features/sentiment_compound/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:×
)dense_features/sentiment_compound/ReshapeReshape5dense_features/sentiment_compound/ExpandDims:output:08dense_features/sentiment_compound/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
0dense_features/sentiment_negative/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Â
,dense_features/sentiment_negative/ExpandDims
ExpandDimsinputs_sentiment_negative9dense_features/sentiment_negative/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'dense_features/sentiment_negative/ShapeShape5dense_features/sentiment_negative/ExpandDims:output:0*
T0*
_output_shapes
:
5dense_features/sentiment_negative/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7dense_features/sentiment_negative/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7dense_features/sentiment_negative/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ű
/dense_features/sentiment_negative/strided_sliceStridedSlice0dense_features/sentiment_negative/Shape:output:0>dense_features/sentiment_negative/strided_slice/stack:output:0@dense_features/sentiment_negative/strided_slice/stack_1:output:0@dense_features/sentiment_negative/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features/sentiment_negative/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ű
/dense_features/sentiment_negative/Reshape/shapePack8dense_features/sentiment_negative/strided_slice:output:0:dense_features/sentiment_negative/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:×
)dense_features/sentiment_negative/ReshapeReshape5dense_features/sentiment_negative/ExpandDims:output:08dense_features/sentiment_negative/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙{
0dense_features/sentiment_positive/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Â
,dense_features/sentiment_positive/ExpandDims
ExpandDimsinputs_sentiment_positive9dense_features/sentiment_positive/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
'dense_features/sentiment_positive/ShapeShape5dense_features/sentiment_positive/ExpandDims:output:0*
T0*
_output_shapes
:
5dense_features/sentiment_positive/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
7dense_features/sentiment_positive/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
7dense_features/sentiment_positive/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ű
/dense_features/sentiment_positive/strided_sliceStridedSlice0dense_features/sentiment_positive/Shape:output:0>dense_features/sentiment_positive/strided_slice/stack:output:0@dense_features/sentiment_positive/strided_slice/stack_1:output:0@dense_features/sentiment_positive/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1dense_features/sentiment_positive/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ű
/dense_features/sentiment_positive/Reshape/shapePack8dense_features/sentiment_positive/strided_slice:output:0:dense_features/sentiment_positive/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:×
)dense_features/sentiment_positive/ReshapeReshape5dense_features/sentiment_positive/ExpandDims:output:08dense_features/sentiment_positive/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙|
1dense_features/stopwords_frequency/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙Ĺ
-dense_features/stopwords_frequency/ExpandDims
ExpandDimsinputs_stopwords_frequency:dense_features/stopwords_frequency/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
(dense_features/stopwords_frequency/ShapeShape6dense_features/stopwords_frequency/ExpandDims:output:0*
T0*
_output_shapes
:
6dense_features/stopwords_frequency/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8dense_features/stopwords_frequency/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8dense_features/stopwords_frequency/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
0dense_features/stopwords_frequency/strided_sliceStridedSlice1dense_features/stopwords_frequency/Shape:output:0?dense_features/stopwords_frequency/strided_slice/stack:output:0Adense_features/stopwords_frequency/strided_slice/stack_1:output:0Adense_features/stopwords_frequency/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2dense_features/stopwords_frequency/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ţ
0dense_features/stopwords_frequency/Reshape/shapePack9dense_features/stopwords_frequency/strided_slice:output:0;dense_features/stopwords_frequency/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ú
*dense_features/stopwords_frequency/ReshapeReshape6dense_features/stopwords_frequency/ExpandDims:output:09dense_features/stopwords_frequency/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙v
+dense_features/text_standard/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙ł
'dense_features/text_standard/ExpandDims
ExpandDimsinputs_text_standard4dense_features/text_standard/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
"dense_features/text_standard/ShapeShape0dense_features/text_standard/ExpandDims:output:0*
T0*
_output_shapes
:z
0dense_features/text_standard/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2dense_features/text_standard/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2dense_features/text_standard/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
*dense_features/text_standard/strided_sliceStridedSlice+dense_features/text_standard/Shape:output:09dense_features/text_standard/strided_slice/stack:output:0;dense_features/text_standard/strided_slice/stack_1:output:0;dense_features/text_standard/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,dense_features/text_standard/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ě
*dense_features/text_standard/Reshape/shapePack3dense_features/text_standard/strided_slice:output:05dense_features/text_standard/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Č
$dense_features/text_standard/ReshapeReshape0dense_features/text_standard/ExpandDims:output:03dense_features/text_standard/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙l
!dense_features/ttr/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙
dense_features/ttr/ExpandDims
ExpandDims
inputs_ttr*dense_features/ttr/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙n
dense_features/ttr/ShapeShape&dense_features/ttr/ExpandDims:output:0*
T0*
_output_shapes
:p
&dense_features/ttr/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/ttr/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/ttr/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 dense_features/ttr/strided_sliceStridedSlice!dense_features/ttr/Shape:output:0/dense_features/ttr/strided_slice/stack:output:01dense_features/ttr/strided_slice/stack_1:output:01dense_features/ttr/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/ttr/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ž
 dense_features/ttr/Reshape/shapePack)dense_features/ttr/strided_slice:output:0+dense_features/ttr/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ş
dense_features/ttr/ReshapeReshape&dense_features/ttr/ExpandDims:output:0)dense_features/ttr/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙t
)dense_features/verb_to_adv/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙­
%dense_features/verb_to_adv/ExpandDims
ExpandDimsinputs_verb_to_adv2dense_features/verb_to_adv/ExpandDims/dim:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙~
 dense_features/verb_to_adv/ShapeShape.dense_features/verb_to_adv/ExpandDims:output:0*
T0*
_output_shapes
:x
.dense_features/verb_to_adv/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0dense_features/verb_to_adv/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0dense_features/verb_to_adv/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ř
(dense_features/verb_to_adv/strided_sliceStridedSlice)dense_features/verb_to_adv/Shape:output:07dense_features/verb_to_adv/strided_slice/stack:output:09dense_features/verb_to_adv/strided_slice/stack_1:output:09dense_features/verb_to_adv/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*dense_features/verb_to_adv/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ć
(dense_features/verb_to_adv/Reshape/shapePack1dense_features/verb_to_adv/strided_slice:output:03dense_features/verb_to_adv/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Â
"dense_features/verb_to_adv/ReshapeReshape.dense_features/verb_to_adv/ExpandDims:output:01dense_features/verb_to_adv/Reshape/shape:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
˙˙˙˙˙˙˙˙˙×
dense_features/concatConcatV2#dense_features/ARI/Reshape:output:04dense_features/Incorrect_form_ratio/Reshape:output:0/dense_features/av_word_per_sen/Reshape:output:0/dense_features/coherence_score/Reshape:output:0<dense_features/dale_chall_readability_score/Reshape:output:04dense_features/flesch_kincaid_grade/Reshape:output:03dense_features/flesch_reading_ease/Reshape:output:0/dense_features/freq_diff_words/Reshape:output:0+dense_features/freq_of_adj/Reshape:output:0+dense_features/freq_of_adv/Reshape:output:04dense_features/freq_of_distinct_adj/Reshape:output:04dense_features/freq_of_distinct_adv/Reshape:output:0,dense_features/freq_of_noun/Reshape:output:0/dense_features/freq_of_pronoun/Reshape:output:02dense_features/freq_of_transition/Reshape:output:0,dense_features/freq_of_verb/Reshape:output:03dense_features/freq_of_wrong_words/Reshape:output:04dense_features/lexrank_avg_min_diff/Reshape:output:05dense_features/lexrank_interquartile/Reshape:output:0.dense_features/mcalpine_eflaw/Reshape:output:0+dense_features/noun_to_adj/Reshape:output:05dense_features/num_of_grammar_errors/Reshape:output:02dense_features/num_of_short_forms/Reshape:output:04dense_features/number_of_diff_words/Reshape:output:0/dense_features/number_of_words/Reshape:output:00dense_features/phrase_diversity/Reshape:output:0,dense_features/punctuations/Reshape:output:03dense_features/sentence_complexity/Reshape:output:02dense_features/sentiment_compound/Reshape:output:02dense_features/sentiment_negative/Reshape:output:02dense_features/sentiment_positive/Reshape:output:03dense_features/stopwords_frequency/Reshape:output:0-dense_features/text_standard/Reshape:output:0#dense_features/ttr/Reshape:output:0+dense_features/verb_to_adv/Reshape:output:0#dense_features/concat/axis:output:0*
N#*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙#
Hidden0/MatMul/ReadVariableOpReadVariableOp&hidden0_matmul_readvariableop_resource*
_output_shapes

:#*
dtype0
Hidden0/MatMulMatMuldense_features/concat:output:0%Hidden0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Hidden0/BiasAdd/ReadVariableOpReadVariableOp'hidden0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Hidden0/BiasAddBiasAddHidden0/MatMul:product:0&Hidden0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
Hidden0/ReluReluHidden0/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Hidden1/MatMul/ReadVariableOpReadVariableOp&hidden1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
Hidden1/MatMulMatMulHidden0/Relu:activations:0%Hidden1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Hidden1/BiasAdd/ReadVariableOpReadVariableOp'hidden1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Hidden1/BiasAddBiasAddHidden1/MatMul:product:0&Hidden1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙`
Hidden1/ReluReluHidden1/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
Output/MatMulMatMulHidden1/Relu:activations:0$Output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
Output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Output/BiasAddBiasAddOutput/MatMul:product:0%Output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙f
IdentityIdentityOutput/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
NoOpNoOp^Hidden0/BiasAdd/ReadVariableOp^Hidden0/MatMul/ReadVariableOp^Hidden1/BiasAdd/ReadVariableOp^Hidden1/MatMul/ReadVariableOp^Output/BiasAdd/ReadVariableOp^Output/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Ě
_input_shapesş
ˇ:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙: : : : : : 2@
Hidden0/BiasAdd/ReadVariableOpHidden0/BiasAdd/ReadVariableOp2>
Hidden0/MatMul/ReadVariableOpHidden0/MatMul/ReadVariableOp2@
Hidden1/BiasAdd/ReadVariableOpHidden1/BiasAdd/ReadVariableOp2>
Hidden1/MatMul/ReadVariableOpHidden1/MatMul/ReadVariableOp2>
Output/BiasAdd/ReadVariableOpOutput/BiasAdd/ReadVariableOp2<
Output/MatMul/ReadVariableOpOutput/MatMul/ReadVariableOp:O K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs/ARI:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/Incorrect_form_ratio:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/av_word_per_sen:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/coherence_score:TP
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
)
_user_specified_nameinputs/cohesion:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs/corrected_text:hd
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
=
_user_specified_name%#inputs/dale_chall_readability_score:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/flesch_kincaid_grade:_[
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/flesch_reading_ease:[	W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/freq_diff_words:W
S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/freq_of_adj:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/freq_of_adv:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/freq_of_distinct_adj:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/freq_of_distinct_adv:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs/freq_of_noun:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/freq_of_pronoun:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/freq_of_transition:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs/freq_of_verb:_[
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/freq_of_wrong_words:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/lexrank_avg_min_diff:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_nameinputs/lexrank_interquartile:ZV
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
/
_user_specified_nameinputs/mcalpine_eflaw:WS
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/noun_to_adj:a]
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
6
_user_specified_nameinputs/num_of_grammar_errors:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/num_of_short_forms:`\
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
5
_user_specified_nameinputs/number_of_diff_words:[W
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
0
_user_specified_nameinputs/number_of_words:\X
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
1
_user_specified_nameinputs/phrase_diversity:XT
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
-
_user_specified_nameinputs/punctuations:_[
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/sentence_complexity:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/sentiment_compound:^Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/sentiment_negative:^ Z
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
3
_user_specified_nameinputs/sentiment_positive:_![
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
4
_user_specified_nameinputs/stopwords_frequency:Y"U
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
.
_user_specified_nameinputs/text_standard:O#K
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
$
_user_specified_name
inputs/ttr:W$S
#
_output_shapes
:˙˙˙˙˙˙˙˙˙
,
_user_specified_nameinputs/verb_to_adv"żL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ą
serving_default
/
ARI(
serving_default_ARI:0	˙˙˙˙˙˙˙˙˙
Q
Incorrect_form_ratio9
&serving_default_Incorrect_form_ratio:0˙˙˙˙˙˙˙˙˙
G
av_word_per_sen4
!serving_default_av_word_per_sen:0˙˙˙˙˙˙˙˙˙
G
coherence_score4
!serving_default_coherence_score:0˙˙˙˙˙˙˙˙˙
9
cohesion-
serving_default_cohesion:0˙˙˙˙˙˙˙˙˙
E
corrected_text3
 serving_default_corrected_text:0˙˙˙˙˙˙˙˙˙
a
dale_chall_readability_scoreA
.serving_default_dale_chall_readability_score:0˙˙˙˙˙˙˙˙˙
Q
flesch_kincaid_grade9
&serving_default_flesch_kincaid_grade:0˙˙˙˙˙˙˙˙˙
O
flesch_reading_ease8
%serving_default_flesch_reading_ease:0˙˙˙˙˙˙˙˙˙
G
freq_diff_words4
!serving_default_freq_diff_words:0˙˙˙˙˙˙˙˙˙
?
freq_of_adj0
serving_default_freq_of_adj:0˙˙˙˙˙˙˙˙˙
?
freq_of_adv0
serving_default_freq_of_adv:0˙˙˙˙˙˙˙˙˙
Q
freq_of_distinct_adj9
&serving_default_freq_of_distinct_adj:0˙˙˙˙˙˙˙˙˙
Q
freq_of_distinct_adv9
&serving_default_freq_of_distinct_adv:0˙˙˙˙˙˙˙˙˙
A
freq_of_noun1
serving_default_freq_of_noun:0˙˙˙˙˙˙˙˙˙
G
freq_of_pronoun4
!serving_default_freq_of_pronoun:0˙˙˙˙˙˙˙˙˙
M
freq_of_transition7
$serving_default_freq_of_transition:0˙˙˙˙˙˙˙˙˙
A
freq_of_verb1
serving_default_freq_of_verb:0˙˙˙˙˙˙˙˙˙
O
freq_of_wrong_words8
%serving_default_freq_of_wrong_words:0˙˙˙˙˙˙˙˙˙
Q
lexrank_avg_min_diff9
&serving_default_lexrank_avg_min_diff:0˙˙˙˙˙˙˙˙˙
S
lexrank_interquartile:
'serving_default_lexrank_interquartile:0˙˙˙˙˙˙˙˙˙
E
mcalpine_eflaw3
 serving_default_mcalpine_eflaw:0˙˙˙˙˙˙˙˙˙
?
noun_to_adj0
serving_default_noun_to_adj:0˙˙˙˙˙˙˙˙˙
S
num_of_grammar_errors:
'serving_default_num_of_grammar_errors:0	˙˙˙˙˙˙˙˙˙
M
num_of_short_forms7
$serving_default_num_of_short_forms:0	˙˙˙˙˙˙˙˙˙
Q
number_of_diff_words9
&serving_default_number_of_diff_words:0	˙˙˙˙˙˙˙˙˙
G
number_of_words4
!serving_default_number_of_words:0	˙˙˙˙˙˙˙˙˙
I
phrase_diversity5
"serving_default_phrase_diversity:0˙˙˙˙˙˙˙˙˙
A
punctuations1
serving_default_punctuations:0˙˙˙˙˙˙˙˙˙
O
sentence_complexity8
%serving_default_sentence_complexity:0˙˙˙˙˙˙˙˙˙
M
sentiment_compound7
$serving_default_sentiment_compound:0˙˙˙˙˙˙˙˙˙
M
sentiment_negative7
$serving_default_sentiment_negative:0˙˙˙˙˙˙˙˙˙
M
sentiment_positive7
$serving_default_sentiment_positive:0˙˙˙˙˙˙˙˙˙
O
stopwords_frequency8
%serving_default_stopwords_frequency:0˙˙˙˙˙˙˙˙˙
C
text_standard2
serving_default_text_standard:0˙˙˙˙˙˙˙˙˙
/
ttr(
serving_default_ttr:0˙˙˙˙˙˙˙˙˙
?
verb_to_adv0
serving_default_verb_to_adv:0˙˙˙˙˙˙˙˙˙<
output_10
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict:Šć

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
_build_input_shape

signatures"
_tf_keras_sequential
Ë
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_feature_columns

_resources"
_tf_keras_layer
ť
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
ť
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
ť
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
J
0
1
%2
&3
-4
.5"
trackable_list_wrapper
J
0
1
%2
&3
-4
.5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ę
/non_trainable_variables

0layers
1metrics
2layer_regularization_losses
3layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
ć
4trace_0
5trace_1
6trace_2
7trace_32ű
,__inference_sequential_2_layer_call_fn_51554
,__inference_sequential_2_layer_call_fn_52505
,__inference_sequential_2_layer_call_fn_52558
,__inference_sequential_2_layer_call_fn_52279Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 z4trace_0z5trace_1z6trace_2z7trace_3
Ň
8trace_0
9trace_1
:trace_2
;trace_32ç
G__inference_sequential_2_layer_call_and_return_conditional_losses_52975
G__inference_sequential_2_layer_call_and_return_conditional_losses_53392
G__inference_sequential_2_layer_call_and_return_conditional_losses_52335
G__inference_sequential_2_layer_call_and_return_conditional_losses_52391Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 z8trace_0z9trace_1z:trace_2z;trace_3
ŇBĎ
 __inference__wrapped_model_51010ARIIncorrect_form_ratioav_word_per_sencoherence_scorecohesioncorrected_textdale_chall_readability_scoreflesch_kincaid_gradeflesch_reading_easefreq_diff_wordsfreq_of_adjfreq_of_advfreq_of_distinct_adjfreq_of_distinct_advfreq_of_nounfreq_of_pronounfreq_of_transitionfreq_of_verbfreq_of_wrong_wordslexrank_avg_min_difflexrank_interquartilemcalpine_eflawnoun_to_adjnum_of_grammar_errorsnum_of_short_formsnumber_of_diff_wordsnumber_of_wordsphrase_diversitypunctuationssentence_complexitysentiment_compoundsentiment_negativesentiment_positivestopwords_frequencytext_standardttrverb_to_adv%"
˛
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ż
<iter

=beta_1

>beta_2
	?decay
@learning_ratemqmr%ms&mt-mu.mvvwvx%vy&vz-v{.v|"
	optimizer
 "
trackable_dict_wrapper
,
Aserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
î
Gtrace_0
Htrace_12ˇ
.__inference_dense_features_layer_call_fn_53433
.__inference_dense_features_layer_call_fn_53474Ô
Ë˛Ç
FullArgSpecE
args=:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 zGtrace_0zHtrace_1
¤
Itrace_0
Jtrace_12í
I__inference_dense_features_layer_call_and_return_conditional_losses_53871
I__inference_dense_features_layer_call_and_return_conditional_losses_54268Ô
Ë˛Ç
FullArgSpecE
args=:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 zItrace_0zJtrace_1
 "
trackable_list_wrapper
"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ë
Ptrace_02Î
'__inference_Hidden0_layer_call_fn_54277˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zPtrace_0

Qtrace_02é
B__inference_Hidden0_layer_call_and_return_conditional_losses_54288˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zQtrace_0
-:+#2sequential_2/Hidden0/kernel
':%2sequential_2/Hidden0/bias
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Rnon_trainable_variables

Slayers
Tmetrics
Ulayer_regularization_losses
Vlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
ë
Wtrace_02Î
'__inference_Hidden1_layer_call_fn_54297˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zWtrace_0

Xtrace_02é
B__inference_Hidden1_layer_call_and_return_conditional_losses_54308˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 zXtrace_0
-:+2sequential_2/Hidden1/kernel
':%2sequential_2/Hidden1/bias
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
ę
^trace_02Í
&__inference_Output_layer_call_fn_54317˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z^trace_0

_trace_02č
A__inference_Output_layer_call_and_return_conditional_losses_54327˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 z_trace_0
,:*2sequential_2/Output/kernel
&:$2sequential_2/Output/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
5
`0
a1
b2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_sequential_2_layer_call_fn_51554ARIIncorrect_form_ratioav_word_per_sencoherence_scorecohesioncorrected_textdale_chall_readability_scoreflesch_kincaid_gradeflesch_reading_easefreq_diff_wordsfreq_of_adjfreq_of_advfreq_of_distinct_adjfreq_of_distinct_advfreq_of_nounfreq_of_pronounfreq_of_transitionfreq_of_verbfreq_of_wrong_wordslexrank_avg_min_difflexrank_interquartilemcalpine_eflawnoun_to_adjnum_of_grammar_errorsnum_of_short_formsnumber_of_diff_wordsnumber_of_wordsphrase_diversitypunctuationssentence_complexitysentiment_compoundsentiment_negativesentiment_positivestopwords_frequencytext_standardttrverb_to_adv%"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
	B	
,__inference_sequential_2_layer_call_fn_52505
inputs/ARIinputs/Incorrect_form_ratioinputs/av_word_per_seninputs/coherence_scoreinputs/cohesioninputs/corrected_text#inputs/dale_chall_readability_scoreinputs/flesch_kincaid_gradeinputs/flesch_reading_easeinputs/freq_diff_wordsinputs/freq_of_adjinputs/freq_of_advinputs/freq_of_distinct_adjinputs/freq_of_distinct_advinputs/freq_of_nouninputs/freq_of_pronouninputs/freq_of_transitioninputs/freq_of_verbinputs/freq_of_wrong_wordsinputs/lexrank_avg_min_diffinputs/lexrank_interquartileinputs/mcalpine_eflawinputs/noun_to_adjinputs/num_of_grammar_errorsinputs/num_of_short_formsinputs/number_of_diff_wordsinputs/number_of_wordsinputs/phrase_diversityinputs/punctuationsinputs/sentence_complexityinputs/sentiment_compoundinputs/sentiment_negativeinputs/sentiment_positiveinputs/stopwords_frequencyinputs/text_standard
inputs/ttrinputs/verb_to_adv%"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
	B	
,__inference_sequential_2_layer_call_fn_52558
inputs/ARIinputs/Incorrect_form_ratioinputs/av_word_per_seninputs/coherence_scoreinputs/cohesioninputs/corrected_text#inputs/dale_chall_readability_scoreinputs/flesch_kincaid_gradeinputs/flesch_reading_easeinputs/freq_diff_wordsinputs/freq_of_adjinputs/freq_of_advinputs/freq_of_distinct_adjinputs/freq_of_distinct_advinputs/freq_of_nouninputs/freq_of_pronouninputs/freq_of_transitioninputs/freq_of_verbinputs/freq_of_wrong_wordsinputs/lexrank_avg_min_diffinputs/lexrank_interquartileinputs/mcalpine_eflawinputs/noun_to_adjinputs/num_of_grammar_errorsinputs/num_of_short_formsinputs/number_of_diff_wordsinputs/number_of_wordsinputs/phrase_diversityinputs/punctuationsinputs/sentence_complexityinputs/sentiment_compoundinputs/sentiment_negativeinputs/sentiment_positiveinputs/stopwords_frequencyinputs/text_standard
inputs/ttrinputs/verb_to_adv%"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
B
,__inference_sequential_2_layer_call_fn_52279ARIIncorrect_form_ratioav_word_per_sencoherence_scorecohesioncorrected_textdale_chall_readability_scoreflesch_kincaid_gradeflesch_reading_easefreq_diff_wordsfreq_of_adjfreq_of_advfreq_of_distinct_adjfreq_of_distinct_advfreq_of_nounfreq_of_pronounfreq_of_transitionfreq_of_verbfreq_of_wrong_wordslexrank_avg_min_difflexrank_interquartilemcalpine_eflawnoun_to_adjnum_of_grammar_errorsnum_of_short_formsnumber_of_diff_wordsnumber_of_wordsphrase_diversitypunctuationssentence_complexitysentiment_compoundsentiment_negativesentiment_positivestopwords_frequencytext_standardttrverb_to_adv%"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
¤	BĄ	
G__inference_sequential_2_layer_call_and_return_conditional_losses_52975
inputs/ARIinputs/Incorrect_form_ratioinputs/av_word_per_seninputs/coherence_scoreinputs/cohesioninputs/corrected_text#inputs/dale_chall_readability_scoreinputs/flesch_kincaid_gradeinputs/flesch_reading_easeinputs/freq_diff_wordsinputs/freq_of_adjinputs/freq_of_advinputs/freq_of_distinct_adjinputs/freq_of_distinct_advinputs/freq_of_nouninputs/freq_of_pronouninputs/freq_of_transitioninputs/freq_of_verbinputs/freq_of_wrong_wordsinputs/lexrank_avg_min_diffinputs/lexrank_interquartileinputs/mcalpine_eflawinputs/noun_to_adjinputs/num_of_grammar_errorsinputs/num_of_short_formsinputs/number_of_diff_wordsinputs/number_of_wordsinputs/phrase_diversityinputs/punctuationsinputs/sentence_complexityinputs/sentiment_compoundinputs/sentiment_negativeinputs/sentiment_positiveinputs/stopwords_frequencyinputs/text_standard
inputs/ttrinputs/verb_to_adv%"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
¤	BĄ	
G__inference_sequential_2_layer_call_and_return_conditional_losses_53392
inputs/ARIinputs/Incorrect_form_ratioinputs/av_word_per_seninputs/coherence_scoreinputs/cohesioninputs/corrected_text#inputs/dale_chall_readability_scoreinputs/flesch_kincaid_gradeinputs/flesch_reading_easeinputs/freq_diff_wordsinputs/freq_of_adjinputs/freq_of_advinputs/freq_of_distinct_adjinputs/freq_of_distinct_advinputs/freq_of_nouninputs/freq_of_pronouninputs/freq_of_transitioninputs/freq_of_verbinputs/freq_of_wrong_wordsinputs/lexrank_avg_min_diffinputs/lexrank_interquartileinputs/mcalpine_eflawinputs/noun_to_adjinputs/num_of_grammar_errorsinputs/num_of_short_formsinputs/number_of_diff_wordsinputs/number_of_wordsinputs/phrase_diversityinputs/punctuationsinputs/sentence_complexityinputs/sentiment_compoundinputs/sentiment_negativeinputs/sentiment_positiveinputs/stopwords_frequencyinputs/text_standard
inputs/ttrinputs/verb_to_adv%"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ĄB
G__inference_sequential_2_layer_call_and_return_conditional_losses_52335ARIIncorrect_form_ratioav_word_per_sencoherence_scorecohesioncorrected_textdale_chall_readability_scoreflesch_kincaid_gradeflesch_reading_easefreq_diff_wordsfreq_of_adjfreq_of_advfreq_of_distinct_adjfreq_of_distinct_advfreq_of_nounfreq_of_pronounfreq_of_transitionfreq_of_verbfreq_of_wrong_wordslexrank_avg_min_difflexrank_interquartilemcalpine_eflawnoun_to_adjnum_of_grammar_errorsnum_of_short_formsnumber_of_diff_wordsnumber_of_wordsphrase_diversitypunctuationssentence_complexitysentiment_compoundsentiment_negativesentiment_positivestopwords_frequencytext_standardttrverb_to_adv%"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ĄB
G__inference_sequential_2_layer_call_and_return_conditional_losses_52391ARIIncorrect_form_ratioav_word_per_sencoherence_scorecohesioncorrected_textdale_chall_readability_scoreflesch_kincaid_gradeflesch_reading_easefreq_diff_wordsfreq_of_adjfreq_of_advfreq_of_distinct_adjfreq_of_distinct_advfreq_of_nounfreq_of_pronounfreq_of_transitionfreq_of_verbfreq_of_wrong_wordslexrank_avg_min_difflexrank_interquartilemcalpine_eflawnoun_to_adjnum_of_grammar_errorsnum_of_short_formsnumber_of_diff_wordsnumber_of_wordsphrase_diversitypunctuationssentence_complexitysentiment_compoundsentiment_negativesentiment_positivestopwords_frequencytext_standardttrverb_to_adv%"Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ĎBĚ
#__inference_signature_wrapper_52452ARIIncorrect_form_ratioav_word_per_sencoherence_scorecohesioncorrected_textdale_chall_readability_scoreflesch_kincaid_gradeflesch_reading_easefreq_diff_wordsfreq_of_adjfreq_of_advfreq_of_distinct_adjfreq_of_distinct_advfreq_of_nounfreq_of_pronounfreq_of_transitionfreq_of_verbfreq_of_wrong_wordslexrank_avg_min_difflexrank_interquartilemcalpine_eflawnoun_to_adjnum_of_grammar_errorsnum_of_short_formsnumber_of_diff_wordsnumber_of_wordsphrase_diversitypunctuationssentence_complexitysentiment_compoundsentiment_negativesentiment_positivestopwords_frequencytext_standardttrverb_to_adv"
˛
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
é	Bć	
.__inference_dense_features_layer_call_fn_53433features/ARIfeatures/Incorrect_form_ratiofeatures/av_word_per_senfeatures/coherence_scorefeatures/cohesionfeatures/corrected_text%features/dale_chall_readability_scorefeatures/flesch_kincaid_gradefeatures/flesch_reading_easefeatures/freq_diff_wordsfeatures/freq_of_adjfeatures/freq_of_advfeatures/freq_of_distinct_adjfeatures/freq_of_distinct_advfeatures/freq_of_nounfeatures/freq_of_pronounfeatures/freq_of_transitionfeatures/freq_of_verbfeatures/freq_of_wrong_wordsfeatures/lexrank_avg_min_difffeatures/lexrank_interquartilefeatures/mcalpine_eflawfeatures/noun_to_adjfeatures/num_of_grammar_errorsfeatures/num_of_short_formsfeatures/number_of_diff_wordsfeatures/number_of_wordsfeatures/phrase_diversityfeatures/punctuationsfeatures/sentence_complexityfeatures/sentiment_compoundfeatures/sentiment_negativefeatures/sentiment_positivefeatures/stopwords_frequencyfeatures/text_standardfeatures/ttrfeatures/verb_to_adv%"Ô
Ë˛Ç
FullArgSpecE
args=:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
é	Bć	
.__inference_dense_features_layer_call_fn_53474features/ARIfeatures/Incorrect_form_ratiofeatures/av_word_per_senfeatures/coherence_scorefeatures/cohesionfeatures/corrected_text%features/dale_chall_readability_scorefeatures/flesch_kincaid_gradefeatures/flesch_reading_easefeatures/freq_diff_wordsfeatures/freq_of_adjfeatures/freq_of_advfeatures/freq_of_distinct_adjfeatures/freq_of_distinct_advfeatures/freq_of_nounfeatures/freq_of_pronounfeatures/freq_of_transitionfeatures/freq_of_verbfeatures/freq_of_wrong_wordsfeatures/lexrank_avg_min_difffeatures/lexrank_interquartilefeatures/mcalpine_eflawfeatures/noun_to_adjfeatures/num_of_grammar_errorsfeatures/num_of_short_formsfeatures/number_of_diff_wordsfeatures/number_of_wordsfeatures/phrase_diversityfeatures/punctuationsfeatures/sentence_complexityfeatures/sentiment_compoundfeatures/sentiment_negativefeatures/sentiment_positivefeatures/stopwords_frequencyfeatures/text_standardfeatures/ttrfeatures/verb_to_adv%"Ô
Ë˛Ç
FullArgSpecE
args=:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 

B

I__inference_dense_features_layer_call_and_return_conditional_losses_53871features/ARIfeatures/Incorrect_form_ratiofeatures/av_word_per_senfeatures/coherence_scorefeatures/cohesionfeatures/corrected_text%features/dale_chall_readability_scorefeatures/flesch_kincaid_gradefeatures/flesch_reading_easefeatures/freq_diff_wordsfeatures/freq_of_adjfeatures/freq_of_advfeatures/freq_of_distinct_adjfeatures/freq_of_distinct_advfeatures/freq_of_nounfeatures/freq_of_pronounfeatures/freq_of_transitionfeatures/freq_of_verbfeatures/freq_of_wrong_wordsfeatures/lexrank_avg_min_difffeatures/lexrank_interquartilefeatures/mcalpine_eflawfeatures/noun_to_adjfeatures/num_of_grammar_errorsfeatures/num_of_short_formsfeatures/number_of_diff_wordsfeatures/number_of_wordsfeatures/phrase_diversityfeatures/punctuationsfeatures/sentence_complexityfeatures/sentiment_compoundfeatures/sentiment_negativefeatures/sentiment_positivefeatures/stopwords_frequencyfeatures/text_standardfeatures/ttrfeatures/verb_to_adv%"Ô
Ë˛Ç
FullArgSpecE
args=:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 

B

I__inference_dense_features_layer_call_and_return_conditional_losses_54268features/ARIfeatures/Incorrect_form_ratiofeatures/av_word_per_senfeatures/coherence_scorefeatures/cohesionfeatures/corrected_text%features/dale_chall_readability_scorefeatures/flesch_kincaid_gradefeatures/flesch_reading_easefeatures/freq_diff_wordsfeatures/freq_of_adjfeatures/freq_of_advfeatures/freq_of_distinct_adjfeatures/freq_of_distinct_advfeatures/freq_of_nounfeatures/freq_of_pronounfeatures/freq_of_transitionfeatures/freq_of_verbfeatures/freq_of_wrong_wordsfeatures/lexrank_avg_min_difffeatures/lexrank_interquartilefeatures/mcalpine_eflawfeatures/noun_to_adjfeatures/num_of_grammar_errorsfeatures/num_of_short_formsfeatures/number_of_diff_wordsfeatures/number_of_wordsfeatures/phrase_diversityfeatures/punctuationsfeatures/sentence_complexityfeatures/sentiment_compoundfeatures/sentiment_negativefeatures/sentiment_positivefeatures/stopwords_frequencyfeatures/text_standardfeatures/ttrfeatures/verb_to_adv%"Ô
Ë˛Ç
FullArgSpecE
args=:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaults

 
p 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŰBŘ
'__inference_Hidden0_layer_call_fn_54277inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
öBó
B__inference_Hidden0_layer_call_and_return_conditional_losses_54288inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ŰBŘ
'__inference_Hidden1_layer_call_fn_54297inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
öBó
B__inference_Hidden1_layer_call_and_return_conditional_losses_54308inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÚB×
&__inference_Output_layer_call_fn_54317inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
őBň
A__inference_Output_layer_call_and_return_conditional_losses_54327inputs"˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
N
c	variables
d	keras_api
	etotal
	fcount"
_tf_keras_metric
^
g	variables
h	keras_api
	itotal
	jcount
k
_fn_kwargs"
_tf_keras_metric
^
l	variables
m	keras_api
	ntotal
	ocount
p
_fn_kwargs"
_tf_keras_metric
.
e0
f1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
.
i0
j1"
trackable_list_wrapper
-
g	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
n0
o1"
trackable_list_wrapper
-
l	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
2:0#2"Adam/sequential_2/Hidden0/kernel/m
,:*2 Adam/sequential_2/Hidden0/bias/m
2:02"Adam/sequential_2/Hidden1/kernel/m
,:*2 Adam/sequential_2/Hidden1/bias/m
1:/2!Adam/sequential_2/Output/kernel/m
+:)2Adam/sequential_2/Output/bias/m
2:0#2"Adam/sequential_2/Hidden0/kernel/v
,:*2 Adam/sequential_2/Hidden0/bias/v
2:02"Adam/sequential_2/Hidden1/kernel/v
,:*2 Adam/sequential_2/Hidden1/bias/v
1:/2!Adam/sequential_2/Output/kernel/v
+:)2Adam/sequential_2/Output/bias/v˘
B__inference_Hidden0_layer_call_and_return_conditional_losses_54288\/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙#
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 z
'__inference_Hidden0_layer_call_fn_54277O/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙#
Ş "˙˙˙˙˙˙˙˙˙˘
B__inference_Hidden1_layer_call_and_return_conditional_losses_54308\%&/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 z
'__inference_Hidden1_layer_call_fn_54297O%&/˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ą
A__inference_Output_layer_call_and_return_conditional_losses_54327\-./˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 y
&__inference_Output_layer_call_fn_54317O-./˘,
%˘"
 
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙
 __inference__wrapped_model_51010ď%&-.Ż˘Ť
Ł˘
Ş
 
ARI
ARI˙˙˙˙˙˙˙˙˙	
B
Incorrect_form_ratio*'
Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
8
av_word_per_sen%"
av_word_per_sen˙˙˙˙˙˙˙˙˙
8
coherence_score%"
coherence_score˙˙˙˙˙˙˙˙˙
*
cohesion
cohesion˙˙˙˙˙˙˙˙˙
6
corrected_text$!
corrected_text˙˙˙˙˙˙˙˙˙
R
dale_chall_readability_score2/
dale_chall_readability_score˙˙˙˙˙˙˙˙˙
B
flesch_kincaid_grade*'
flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
@
flesch_reading_ease)&
flesch_reading_ease˙˙˙˙˙˙˙˙˙
8
freq_diff_words%"
freq_diff_words˙˙˙˙˙˙˙˙˙
0
freq_of_adj!
freq_of_adj˙˙˙˙˙˙˙˙˙
0
freq_of_adv!
freq_of_adv˙˙˙˙˙˙˙˙˙
B
freq_of_distinct_adj*'
freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
B
freq_of_distinct_adv*'
freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
2
freq_of_noun"
freq_of_noun˙˙˙˙˙˙˙˙˙
8
freq_of_pronoun%"
freq_of_pronoun˙˙˙˙˙˙˙˙˙
>
freq_of_transition(%
freq_of_transition˙˙˙˙˙˙˙˙˙
2
freq_of_verb"
freq_of_verb˙˙˙˙˙˙˙˙˙
@
freq_of_wrong_words)&
freq_of_wrong_words˙˙˙˙˙˙˙˙˙
B
lexrank_avg_min_diff*'
lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
D
lexrank_interquartile+(
lexrank_interquartile˙˙˙˙˙˙˙˙˙
6
mcalpine_eflaw$!
mcalpine_eflaw˙˙˙˙˙˙˙˙˙
0
noun_to_adj!
noun_to_adj˙˙˙˙˙˙˙˙˙
D
num_of_grammar_errors+(
num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
>
num_of_short_forms(%
num_of_short_forms˙˙˙˙˙˙˙˙˙	
B
number_of_diff_words*'
number_of_diff_words˙˙˙˙˙˙˙˙˙	
8
number_of_words%"
number_of_words˙˙˙˙˙˙˙˙˙	
:
phrase_diversity&#
phrase_diversity˙˙˙˙˙˙˙˙˙
2
punctuations"
punctuations˙˙˙˙˙˙˙˙˙
@
sentence_complexity)&
sentence_complexity˙˙˙˙˙˙˙˙˙
>
sentiment_compound(%
sentiment_compound˙˙˙˙˙˙˙˙˙
>
sentiment_negative(%
sentiment_negative˙˙˙˙˙˙˙˙˙
>
sentiment_positive(%
sentiment_positive˙˙˙˙˙˙˙˙˙
@
stopwords_frequency)&
stopwords_frequency˙˙˙˙˙˙˙˙˙
4
text_standard# 
text_standard˙˙˙˙˙˙˙˙˙
 
ttr
ttr˙˙˙˙˙˙˙˙˙
0
verb_to_adv!
verb_to_adv˙˙˙˙˙˙˙˙˙
Ş "3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙ü
I__inference_dense_features_layer_call_and_return_conditional_losses_53871Ž˘
ř˘ô
éŞĺ
)
ARI"
features/ARI˙˙˙˙˙˙˙˙˙	
K
Incorrect_form_ratio30
features/Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
A
av_word_per_sen.+
features/av_word_per_sen˙˙˙˙˙˙˙˙˙
A
coherence_score.+
features/coherence_score˙˙˙˙˙˙˙˙˙
3
cohesion'$
features/cohesion˙˙˙˙˙˙˙˙˙
?
corrected_text-*
features/corrected_text˙˙˙˙˙˙˙˙˙
[
dale_chall_readability_score;8
%features/dale_chall_readability_score˙˙˙˙˙˙˙˙˙
K
flesch_kincaid_grade30
features/flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
I
flesch_reading_ease2/
features/flesch_reading_ease˙˙˙˙˙˙˙˙˙
A
freq_diff_words.+
features/freq_diff_words˙˙˙˙˙˙˙˙˙
9
freq_of_adj*'
features/freq_of_adj˙˙˙˙˙˙˙˙˙
9
freq_of_adv*'
features/freq_of_adv˙˙˙˙˙˙˙˙˙
K
freq_of_distinct_adj30
features/freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
K
freq_of_distinct_adv30
features/freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
;
freq_of_noun+(
features/freq_of_noun˙˙˙˙˙˙˙˙˙
A
freq_of_pronoun.+
features/freq_of_pronoun˙˙˙˙˙˙˙˙˙
G
freq_of_transition1.
features/freq_of_transition˙˙˙˙˙˙˙˙˙
;
freq_of_verb+(
features/freq_of_verb˙˙˙˙˙˙˙˙˙
I
freq_of_wrong_words2/
features/freq_of_wrong_words˙˙˙˙˙˙˙˙˙
K
lexrank_avg_min_diff30
features/lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
M
lexrank_interquartile41
features/lexrank_interquartile˙˙˙˙˙˙˙˙˙
?
mcalpine_eflaw-*
features/mcalpine_eflaw˙˙˙˙˙˙˙˙˙
9
noun_to_adj*'
features/noun_to_adj˙˙˙˙˙˙˙˙˙
M
num_of_grammar_errors41
features/num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
G
num_of_short_forms1.
features/num_of_short_forms˙˙˙˙˙˙˙˙˙	
K
number_of_diff_words30
features/number_of_diff_words˙˙˙˙˙˙˙˙˙	
A
number_of_words.+
features/number_of_words˙˙˙˙˙˙˙˙˙	
C
phrase_diversity/,
features/phrase_diversity˙˙˙˙˙˙˙˙˙
;
punctuations+(
features/punctuations˙˙˙˙˙˙˙˙˙
I
sentence_complexity2/
features/sentence_complexity˙˙˙˙˙˙˙˙˙
G
sentiment_compound1.
features/sentiment_compound˙˙˙˙˙˙˙˙˙
G
sentiment_negative1.
features/sentiment_negative˙˙˙˙˙˙˙˙˙
G
sentiment_positive1.
features/sentiment_positive˙˙˙˙˙˙˙˙˙
I
stopwords_frequency2/
features/stopwords_frequency˙˙˙˙˙˙˙˙˙
=
text_standard,)
features/text_standard˙˙˙˙˙˙˙˙˙
)
ttr"
features/ttr˙˙˙˙˙˙˙˙˙
9
verb_to_adv*'
features/verb_to_adv˙˙˙˙˙˙˙˙˙

 
p 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙#
 ü
I__inference_dense_features_layer_call_and_return_conditional_losses_54268Ž˘
ř˘ô
éŞĺ
)
ARI"
features/ARI˙˙˙˙˙˙˙˙˙	
K
Incorrect_form_ratio30
features/Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
A
av_word_per_sen.+
features/av_word_per_sen˙˙˙˙˙˙˙˙˙
A
coherence_score.+
features/coherence_score˙˙˙˙˙˙˙˙˙
3
cohesion'$
features/cohesion˙˙˙˙˙˙˙˙˙
?
corrected_text-*
features/corrected_text˙˙˙˙˙˙˙˙˙
[
dale_chall_readability_score;8
%features/dale_chall_readability_score˙˙˙˙˙˙˙˙˙
K
flesch_kincaid_grade30
features/flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
I
flesch_reading_ease2/
features/flesch_reading_ease˙˙˙˙˙˙˙˙˙
A
freq_diff_words.+
features/freq_diff_words˙˙˙˙˙˙˙˙˙
9
freq_of_adj*'
features/freq_of_adj˙˙˙˙˙˙˙˙˙
9
freq_of_adv*'
features/freq_of_adv˙˙˙˙˙˙˙˙˙
K
freq_of_distinct_adj30
features/freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
K
freq_of_distinct_adv30
features/freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
;
freq_of_noun+(
features/freq_of_noun˙˙˙˙˙˙˙˙˙
A
freq_of_pronoun.+
features/freq_of_pronoun˙˙˙˙˙˙˙˙˙
G
freq_of_transition1.
features/freq_of_transition˙˙˙˙˙˙˙˙˙
;
freq_of_verb+(
features/freq_of_verb˙˙˙˙˙˙˙˙˙
I
freq_of_wrong_words2/
features/freq_of_wrong_words˙˙˙˙˙˙˙˙˙
K
lexrank_avg_min_diff30
features/lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
M
lexrank_interquartile41
features/lexrank_interquartile˙˙˙˙˙˙˙˙˙
?
mcalpine_eflaw-*
features/mcalpine_eflaw˙˙˙˙˙˙˙˙˙
9
noun_to_adj*'
features/noun_to_adj˙˙˙˙˙˙˙˙˙
M
num_of_grammar_errors41
features/num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
G
num_of_short_forms1.
features/num_of_short_forms˙˙˙˙˙˙˙˙˙	
K
number_of_diff_words30
features/number_of_diff_words˙˙˙˙˙˙˙˙˙	
A
number_of_words.+
features/number_of_words˙˙˙˙˙˙˙˙˙	
C
phrase_diversity/,
features/phrase_diversity˙˙˙˙˙˙˙˙˙
;
punctuations+(
features/punctuations˙˙˙˙˙˙˙˙˙
I
sentence_complexity2/
features/sentence_complexity˙˙˙˙˙˙˙˙˙
G
sentiment_compound1.
features/sentiment_compound˙˙˙˙˙˙˙˙˙
G
sentiment_negative1.
features/sentiment_negative˙˙˙˙˙˙˙˙˙
G
sentiment_positive1.
features/sentiment_positive˙˙˙˙˙˙˙˙˙
I
stopwords_frequency2/
features/stopwords_frequency˙˙˙˙˙˙˙˙˙
=
text_standard,)
features/text_standard˙˙˙˙˙˙˙˙˙
)
ttr"
features/ttr˙˙˙˙˙˙˙˙˙
9
verb_to_adv*'
features/verb_to_adv˙˙˙˙˙˙˙˙˙

 
p
Ş "%˘"

0˙˙˙˙˙˙˙˙˙#
 Ô
.__inference_dense_features_layer_call_fn_53433Ą˘
ř˘ô
éŞĺ
)
ARI"
features/ARI˙˙˙˙˙˙˙˙˙	
K
Incorrect_form_ratio30
features/Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
A
av_word_per_sen.+
features/av_word_per_sen˙˙˙˙˙˙˙˙˙
A
coherence_score.+
features/coherence_score˙˙˙˙˙˙˙˙˙
3
cohesion'$
features/cohesion˙˙˙˙˙˙˙˙˙
?
corrected_text-*
features/corrected_text˙˙˙˙˙˙˙˙˙
[
dale_chall_readability_score;8
%features/dale_chall_readability_score˙˙˙˙˙˙˙˙˙
K
flesch_kincaid_grade30
features/flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
I
flesch_reading_ease2/
features/flesch_reading_ease˙˙˙˙˙˙˙˙˙
A
freq_diff_words.+
features/freq_diff_words˙˙˙˙˙˙˙˙˙
9
freq_of_adj*'
features/freq_of_adj˙˙˙˙˙˙˙˙˙
9
freq_of_adv*'
features/freq_of_adv˙˙˙˙˙˙˙˙˙
K
freq_of_distinct_adj30
features/freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
K
freq_of_distinct_adv30
features/freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
;
freq_of_noun+(
features/freq_of_noun˙˙˙˙˙˙˙˙˙
A
freq_of_pronoun.+
features/freq_of_pronoun˙˙˙˙˙˙˙˙˙
G
freq_of_transition1.
features/freq_of_transition˙˙˙˙˙˙˙˙˙
;
freq_of_verb+(
features/freq_of_verb˙˙˙˙˙˙˙˙˙
I
freq_of_wrong_words2/
features/freq_of_wrong_words˙˙˙˙˙˙˙˙˙
K
lexrank_avg_min_diff30
features/lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
M
lexrank_interquartile41
features/lexrank_interquartile˙˙˙˙˙˙˙˙˙
?
mcalpine_eflaw-*
features/mcalpine_eflaw˙˙˙˙˙˙˙˙˙
9
noun_to_adj*'
features/noun_to_adj˙˙˙˙˙˙˙˙˙
M
num_of_grammar_errors41
features/num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
G
num_of_short_forms1.
features/num_of_short_forms˙˙˙˙˙˙˙˙˙	
K
number_of_diff_words30
features/number_of_diff_words˙˙˙˙˙˙˙˙˙	
A
number_of_words.+
features/number_of_words˙˙˙˙˙˙˙˙˙	
C
phrase_diversity/,
features/phrase_diversity˙˙˙˙˙˙˙˙˙
;
punctuations+(
features/punctuations˙˙˙˙˙˙˙˙˙
I
sentence_complexity2/
features/sentence_complexity˙˙˙˙˙˙˙˙˙
G
sentiment_compound1.
features/sentiment_compound˙˙˙˙˙˙˙˙˙
G
sentiment_negative1.
features/sentiment_negative˙˙˙˙˙˙˙˙˙
G
sentiment_positive1.
features/sentiment_positive˙˙˙˙˙˙˙˙˙
I
stopwords_frequency2/
features/stopwords_frequency˙˙˙˙˙˙˙˙˙
=
text_standard,)
features/text_standard˙˙˙˙˙˙˙˙˙
)
ttr"
features/ttr˙˙˙˙˙˙˙˙˙
9
verb_to_adv*'
features/verb_to_adv˙˙˙˙˙˙˙˙˙

 
p 
Ş "˙˙˙˙˙˙˙˙˙#Ô
.__inference_dense_features_layer_call_fn_53474Ą˘
ř˘ô
éŞĺ
)
ARI"
features/ARI˙˙˙˙˙˙˙˙˙	
K
Incorrect_form_ratio30
features/Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
A
av_word_per_sen.+
features/av_word_per_sen˙˙˙˙˙˙˙˙˙
A
coherence_score.+
features/coherence_score˙˙˙˙˙˙˙˙˙
3
cohesion'$
features/cohesion˙˙˙˙˙˙˙˙˙
?
corrected_text-*
features/corrected_text˙˙˙˙˙˙˙˙˙
[
dale_chall_readability_score;8
%features/dale_chall_readability_score˙˙˙˙˙˙˙˙˙
K
flesch_kincaid_grade30
features/flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
I
flesch_reading_ease2/
features/flesch_reading_ease˙˙˙˙˙˙˙˙˙
A
freq_diff_words.+
features/freq_diff_words˙˙˙˙˙˙˙˙˙
9
freq_of_adj*'
features/freq_of_adj˙˙˙˙˙˙˙˙˙
9
freq_of_adv*'
features/freq_of_adv˙˙˙˙˙˙˙˙˙
K
freq_of_distinct_adj30
features/freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
K
freq_of_distinct_adv30
features/freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
;
freq_of_noun+(
features/freq_of_noun˙˙˙˙˙˙˙˙˙
A
freq_of_pronoun.+
features/freq_of_pronoun˙˙˙˙˙˙˙˙˙
G
freq_of_transition1.
features/freq_of_transition˙˙˙˙˙˙˙˙˙
;
freq_of_verb+(
features/freq_of_verb˙˙˙˙˙˙˙˙˙
I
freq_of_wrong_words2/
features/freq_of_wrong_words˙˙˙˙˙˙˙˙˙
K
lexrank_avg_min_diff30
features/lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
M
lexrank_interquartile41
features/lexrank_interquartile˙˙˙˙˙˙˙˙˙
?
mcalpine_eflaw-*
features/mcalpine_eflaw˙˙˙˙˙˙˙˙˙
9
noun_to_adj*'
features/noun_to_adj˙˙˙˙˙˙˙˙˙
M
num_of_grammar_errors41
features/num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
G
num_of_short_forms1.
features/num_of_short_forms˙˙˙˙˙˙˙˙˙	
K
number_of_diff_words30
features/number_of_diff_words˙˙˙˙˙˙˙˙˙	
A
number_of_words.+
features/number_of_words˙˙˙˙˙˙˙˙˙	
C
phrase_diversity/,
features/phrase_diversity˙˙˙˙˙˙˙˙˙
;
punctuations+(
features/punctuations˙˙˙˙˙˙˙˙˙
I
sentence_complexity2/
features/sentence_complexity˙˙˙˙˙˙˙˙˙
G
sentiment_compound1.
features/sentiment_compound˙˙˙˙˙˙˙˙˙
G
sentiment_negative1.
features/sentiment_negative˙˙˙˙˙˙˙˙˙
G
sentiment_positive1.
features/sentiment_positive˙˙˙˙˙˙˙˙˙
I
stopwords_frequency2/
features/stopwords_frequency˙˙˙˙˙˙˙˙˙
=
text_standard,)
features/text_standard˙˙˙˙˙˙˙˙˙
)
ttr"
features/ttr˙˙˙˙˙˙˙˙˙
9
verb_to_adv*'
features/verb_to_adv˙˙˙˙˙˙˙˙˙

 
p
Ş "˙˙˙˙˙˙˙˙˙#ľ
G__inference_sequential_2_layer_call_and_return_conditional_losses_52335é%&-.ˇ˘ł
Ť˘§
Ş
 
ARI
ARI˙˙˙˙˙˙˙˙˙	
B
Incorrect_form_ratio*'
Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
8
av_word_per_sen%"
av_word_per_sen˙˙˙˙˙˙˙˙˙
8
coherence_score%"
coherence_score˙˙˙˙˙˙˙˙˙
*
cohesion
cohesion˙˙˙˙˙˙˙˙˙
6
corrected_text$!
corrected_text˙˙˙˙˙˙˙˙˙
R
dale_chall_readability_score2/
dale_chall_readability_score˙˙˙˙˙˙˙˙˙
B
flesch_kincaid_grade*'
flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
@
flesch_reading_ease)&
flesch_reading_ease˙˙˙˙˙˙˙˙˙
8
freq_diff_words%"
freq_diff_words˙˙˙˙˙˙˙˙˙
0
freq_of_adj!
freq_of_adj˙˙˙˙˙˙˙˙˙
0
freq_of_adv!
freq_of_adv˙˙˙˙˙˙˙˙˙
B
freq_of_distinct_adj*'
freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
B
freq_of_distinct_adv*'
freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
2
freq_of_noun"
freq_of_noun˙˙˙˙˙˙˙˙˙
8
freq_of_pronoun%"
freq_of_pronoun˙˙˙˙˙˙˙˙˙
>
freq_of_transition(%
freq_of_transition˙˙˙˙˙˙˙˙˙
2
freq_of_verb"
freq_of_verb˙˙˙˙˙˙˙˙˙
@
freq_of_wrong_words)&
freq_of_wrong_words˙˙˙˙˙˙˙˙˙
B
lexrank_avg_min_diff*'
lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
D
lexrank_interquartile+(
lexrank_interquartile˙˙˙˙˙˙˙˙˙
6
mcalpine_eflaw$!
mcalpine_eflaw˙˙˙˙˙˙˙˙˙
0
noun_to_adj!
noun_to_adj˙˙˙˙˙˙˙˙˙
D
num_of_grammar_errors+(
num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
>
num_of_short_forms(%
num_of_short_forms˙˙˙˙˙˙˙˙˙	
B
number_of_diff_words*'
number_of_diff_words˙˙˙˙˙˙˙˙˙	
8
number_of_words%"
number_of_words˙˙˙˙˙˙˙˙˙	
:
phrase_diversity&#
phrase_diversity˙˙˙˙˙˙˙˙˙
2
punctuations"
punctuations˙˙˙˙˙˙˙˙˙
@
sentence_complexity)&
sentence_complexity˙˙˙˙˙˙˙˙˙
>
sentiment_compound(%
sentiment_compound˙˙˙˙˙˙˙˙˙
>
sentiment_negative(%
sentiment_negative˙˙˙˙˙˙˙˙˙
>
sentiment_positive(%
sentiment_positive˙˙˙˙˙˙˙˙˙
@
stopwords_frequency)&
stopwords_frequency˙˙˙˙˙˙˙˙˙
4
text_standard# 
text_standard˙˙˙˙˙˙˙˙˙
 
ttr
ttr˙˙˙˙˙˙˙˙˙
0
verb_to_adv!
verb_to_adv˙˙˙˙˙˙˙˙˙
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ľ
G__inference_sequential_2_layer_call_and_return_conditional_losses_52391é%&-.ˇ˘ł
Ť˘§
Ş
 
ARI
ARI˙˙˙˙˙˙˙˙˙	
B
Incorrect_form_ratio*'
Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
8
av_word_per_sen%"
av_word_per_sen˙˙˙˙˙˙˙˙˙
8
coherence_score%"
coherence_score˙˙˙˙˙˙˙˙˙
*
cohesion
cohesion˙˙˙˙˙˙˙˙˙
6
corrected_text$!
corrected_text˙˙˙˙˙˙˙˙˙
R
dale_chall_readability_score2/
dale_chall_readability_score˙˙˙˙˙˙˙˙˙
B
flesch_kincaid_grade*'
flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
@
flesch_reading_ease)&
flesch_reading_ease˙˙˙˙˙˙˙˙˙
8
freq_diff_words%"
freq_diff_words˙˙˙˙˙˙˙˙˙
0
freq_of_adj!
freq_of_adj˙˙˙˙˙˙˙˙˙
0
freq_of_adv!
freq_of_adv˙˙˙˙˙˙˙˙˙
B
freq_of_distinct_adj*'
freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
B
freq_of_distinct_adv*'
freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
2
freq_of_noun"
freq_of_noun˙˙˙˙˙˙˙˙˙
8
freq_of_pronoun%"
freq_of_pronoun˙˙˙˙˙˙˙˙˙
>
freq_of_transition(%
freq_of_transition˙˙˙˙˙˙˙˙˙
2
freq_of_verb"
freq_of_verb˙˙˙˙˙˙˙˙˙
@
freq_of_wrong_words)&
freq_of_wrong_words˙˙˙˙˙˙˙˙˙
B
lexrank_avg_min_diff*'
lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
D
lexrank_interquartile+(
lexrank_interquartile˙˙˙˙˙˙˙˙˙
6
mcalpine_eflaw$!
mcalpine_eflaw˙˙˙˙˙˙˙˙˙
0
noun_to_adj!
noun_to_adj˙˙˙˙˙˙˙˙˙
D
num_of_grammar_errors+(
num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
>
num_of_short_forms(%
num_of_short_forms˙˙˙˙˙˙˙˙˙	
B
number_of_diff_words*'
number_of_diff_words˙˙˙˙˙˙˙˙˙	
8
number_of_words%"
number_of_words˙˙˙˙˙˙˙˙˙	
:
phrase_diversity&#
phrase_diversity˙˙˙˙˙˙˙˙˙
2
punctuations"
punctuations˙˙˙˙˙˙˙˙˙
@
sentence_complexity)&
sentence_complexity˙˙˙˙˙˙˙˙˙
>
sentiment_compound(%
sentiment_compound˙˙˙˙˙˙˙˙˙
>
sentiment_negative(%
sentiment_negative˙˙˙˙˙˙˙˙˙
>
sentiment_positive(%
sentiment_positive˙˙˙˙˙˙˙˙˙
@
stopwords_frequency)&
stopwords_frequency˙˙˙˙˙˙˙˙˙
4
text_standard# 
text_standard˙˙˙˙˙˙˙˙˙
 
ttr
ttr˙˙˙˙˙˙˙˙˙
0
verb_to_adv!
verb_to_adv˙˙˙˙˙˙˙˙˙
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ¸
G__inference_sequential_2_layer_call_and_return_conditional_losses_52975ě%&-.ş˘ś
Ž˘Ş
Ş
'
ARI 

inputs/ARI˙˙˙˙˙˙˙˙˙	
I
Incorrect_form_ratio1.
inputs/Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
?
av_word_per_sen,)
inputs/av_word_per_sen˙˙˙˙˙˙˙˙˙
?
coherence_score,)
inputs/coherence_score˙˙˙˙˙˙˙˙˙
1
cohesion%"
inputs/cohesion˙˙˙˙˙˙˙˙˙
=
corrected_text+(
inputs/corrected_text˙˙˙˙˙˙˙˙˙
Y
dale_chall_readability_score96
#inputs/dale_chall_readability_score˙˙˙˙˙˙˙˙˙
I
flesch_kincaid_grade1.
inputs/flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
G
flesch_reading_ease0-
inputs/flesch_reading_ease˙˙˙˙˙˙˙˙˙
?
freq_diff_words,)
inputs/freq_diff_words˙˙˙˙˙˙˙˙˙
7
freq_of_adj(%
inputs/freq_of_adj˙˙˙˙˙˙˙˙˙
7
freq_of_adv(%
inputs/freq_of_adv˙˙˙˙˙˙˙˙˙
I
freq_of_distinct_adj1.
inputs/freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
I
freq_of_distinct_adv1.
inputs/freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
9
freq_of_noun)&
inputs/freq_of_noun˙˙˙˙˙˙˙˙˙
?
freq_of_pronoun,)
inputs/freq_of_pronoun˙˙˙˙˙˙˙˙˙
E
freq_of_transition/,
inputs/freq_of_transition˙˙˙˙˙˙˙˙˙
9
freq_of_verb)&
inputs/freq_of_verb˙˙˙˙˙˙˙˙˙
G
freq_of_wrong_words0-
inputs/freq_of_wrong_words˙˙˙˙˙˙˙˙˙
I
lexrank_avg_min_diff1.
inputs/lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
K
lexrank_interquartile2/
inputs/lexrank_interquartile˙˙˙˙˙˙˙˙˙
=
mcalpine_eflaw+(
inputs/mcalpine_eflaw˙˙˙˙˙˙˙˙˙
7
noun_to_adj(%
inputs/noun_to_adj˙˙˙˙˙˙˙˙˙
K
num_of_grammar_errors2/
inputs/num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
E
num_of_short_forms/,
inputs/num_of_short_forms˙˙˙˙˙˙˙˙˙	
I
number_of_diff_words1.
inputs/number_of_diff_words˙˙˙˙˙˙˙˙˙	
?
number_of_words,)
inputs/number_of_words˙˙˙˙˙˙˙˙˙	
A
phrase_diversity-*
inputs/phrase_diversity˙˙˙˙˙˙˙˙˙
9
punctuations)&
inputs/punctuations˙˙˙˙˙˙˙˙˙
G
sentence_complexity0-
inputs/sentence_complexity˙˙˙˙˙˙˙˙˙
E
sentiment_compound/,
inputs/sentiment_compound˙˙˙˙˙˙˙˙˙
E
sentiment_negative/,
inputs/sentiment_negative˙˙˙˙˙˙˙˙˙
E
sentiment_positive/,
inputs/sentiment_positive˙˙˙˙˙˙˙˙˙
G
stopwords_frequency0-
inputs/stopwords_frequency˙˙˙˙˙˙˙˙˙
;
text_standard*'
inputs/text_standard˙˙˙˙˙˙˙˙˙
'
ttr 

inputs/ttr˙˙˙˙˙˙˙˙˙
7
verb_to_adv(%
inputs/verb_to_adv˙˙˙˙˙˙˙˙˙
p 

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 ¸
G__inference_sequential_2_layer_call_and_return_conditional_losses_53392ě%&-.ş˘ś
Ž˘Ş
Ş
'
ARI 

inputs/ARI˙˙˙˙˙˙˙˙˙	
I
Incorrect_form_ratio1.
inputs/Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
?
av_word_per_sen,)
inputs/av_word_per_sen˙˙˙˙˙˙˙˙˙
?
coherence_score,)
inputs/coherence_score˙˙˙˙˙˙˙˙˙
1
cohesion%"
inputs/cohesion˙˙˙˙˙˙˙˙˙
=
corrected_text+(
inputs/corrected_text˙˙˙˙˙˙˙˙˙
Y
dale_chall_readability_score96
#inputs/dale_chall_readability_score˙˙˙˙˙˙˙˙˙
I
flesch_kincaid_grade1.
inputs/flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
G
flesch_reading_ease0-
inputs/flesch_reading_ease˙˙˙˙˙˙˙˙˙
?
freq_diff_words,)
inputs/freq_diff_words˙˙˙˙˙˙˙˙˙
7
freq_of_adj(%
inputs/freq_of_adj˙˙˙˙˙˙˙˙˙
7
freq_of_adv(%
inputs/freq_of_adv˙˙˙˙˙˙˙˙˙
I
freq_of_distinct_adj1.
inputs/freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
I
freq_of_distinct_adv1.
inputs/freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
9
freq_of_noun)&
inputs/freq_of_noun˙˙˙˙˙˙˙˙˙
?
freq_of_pronoun,)
inputs/freq_of_pronoun˙˙˙˙˙˙˙˙˙
E
freq_of_transition/,
inputs/freq_of_transition˙˙˙˙˙˙˙˙˙
9
freq_of_verb)&
inputs/freq_of_verb˙˙˙˙˙˙˙˙˙
G
freq_of_wrong_words0-
inputs/freq_of_wrong_words˙˙˙˙˙˙˙˙˙
I
lexrank_avg_min_diff1.
inputs/lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
K
lexrank_interquartile2/
inputs/lexrank_interquartile˙˙˙˙˙˙˙˙˙
=
mcalpine_eflaw+(
inputs/mcalpine_eflaw˙˙˙˙˙˙˙˙˙
7
noun_to_adj(%
inputs/noun_to_adj˙˙˙˙˙˙˙˙˙
K
num_of_grammar_errors2/
inputs/num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
E
num_of_short_forms/,
inputs/num_of_short_forms˙˙˙˙˙˙˙˙˙	
I
number_of_diff_words1.
inputs/number_of_diff_words˙˙˙˙˙˙˙˙˙	
?
number_of_words,)
inputs/number_of_words˙˙˙˙˙˙˙˙˙	
A
phrase_diversity-*
inputs/phrase_diversity˙˙˙˙˙˙˙˙˙
9
punctuations)&
inputs/punctuations˙˙˙˙˙˙˙˙˙
G
sentence_complexity0-
inputs/sentence_complexity˙˙˙˙˙˙˙˙˙
E
sentiment_compound/,
inputs/sentiment_compound˙˙˙˙˙˙˙˙˙
E
sentiment_negative/,
inputs/sentiment_negative˙˙˙˙˙˙˙˙˙
E
sentiment_positive/,
inputs/sentiment_positive˙˙˙˙˙˙˙˙˙
G
stopwords_frequency0-
inputs/stopwords_frequency˙˙˙˙˙˙˙˙˙
;
text_standard*'
inputs/text_standard˙˙˙˙˙˙˙˙˙
'
ttr 

inputs/ttr˙˙˙˙˙˙˙˙˙
7
verb_to_adv(%
inputs/verb_to_adv˙˙˙˙˙˙˙˙˙
p

 
Ş "%˘"

0˙˙˙˙˙˙˙˙˙
 
,__inference_sequential_2_layer_call_fn_51554Ü%&-.ˇ˘ł
Ť˘§
Ş
 
ARI
ARI˙˙˙˙˙˙˙˙˙	
B
Incorrect_form_ratio*'
Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
8
av_word_per_sen%"
av_word_per_sen˙˙˙˙˙˙˙˙˙
8
coherence_score%"
coherence_score˙˙˙˙˙˙˙˙˙
*
cohesion
cohesion˙˙˙˙˙˙˙˙˙
6
corrected_text$!
corrected_text˙˙˙˙˙˙˙˙˙
R
dale_chall_readability_score2/
dale_chall_readability_score˙˙˙˙˙˙˙˙˙
B
flesch_kincaid_grade*'
flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
@
flesch_reading_ease)&
flesch_reading_ease˙˙˙˙˙˙˙˙˙
8
freq_diff_words%"
freq_diff_words˙˙˙˙˙˙˙˙˙
0
freq_of_adj!
freq_of_adj˙˙˙˙˙˙˙˙˙
0
freq_of_adv!
freq_of_adv˙˙˙˙˙˙˙˙˙
B
freq_of_distinct_adj*'
freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
B
freq_of_distinct_adv*'
freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
2
freq_of_noun"
freq_of_noun˙˙˙˙˙˙˙˙˙
8
freq_of_pronoun%"
freq_of_pronoun˙˙˙˙˙˙˙˙˙
>
freq_of_transition(%
freq_of_transition˙˙˙˙˙˙˙˙˙
2
freq_of_verb"
freq_of_verb˙˙˙˙˙˙˙˙˙
@
freq_of_wrong_words)&
freq_of_wrong_words˙˙˙˙˙˙˙˙˙
B
lexrank_avg_min_diff*'
lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
D
lexrank_interquartile+(
lexrank_interquartile˙˙˙˙˙˙˙˙˙
6
mcalpine_eflaw$!
mcalpine_eflaw˙˙˙˙˙˙˙˙˙
0
noun_to_adj!
noun_to_adj˙˙˙˙˙˙˙˙˙
D
num_of_grammar_errors+(
num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
>
num_of_short_forms(%
num_of_short_forms˙˙˙˙˙˙˙˙˙	
B
number_of_diff_words*'
number_of_diff_words˙˙˙˙˙˙˙˙˙	
8
number_of_words%"
number_of_words˙˙˙˙˙˙˙˙˙	
:
phrase_diversity&#
phrase_diversity˙˙˙˙˙˙˙˙˙
2
punctuations"
punctuations˙˙˙˙˙˙˙˙˙
@
sentence_complexity)&
sentence_complexity˙˙˙˙˙˙˙˙˙
>
sentiment_compound(%
sentiment_compound˙˙˙˙˙˙˙˙˙
>
sentiment_negative(%
sentiment_negative˙˙˙˙˙˙˙˙˙
>
sentiment_positive(%
sentiment_positive˙˙˙˙˙˙˙˙˙
@
stopwords_frequency)&
stopwords_frequency˙˙˙˙˙˙˙˙˙
4
text_standard# 
text_standard˙˙˙˙˙˙˙˙˙
 
ttr
ttr˙˙˙˙˙˙˙˙˙
0
verb_to_adv!
verb_to_adv˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
,__inference_sequential_2_layer_call_fn_52279Ü%&-.ˇ˘ł
Ť˘§
Ş
 
ARI
ARI˙˙˙˙˙˙˙˙˙	
B
Incorrect_form_ratio*'
Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
8
av_word_per_sen%"
av_word_per_sen˙˙˙˙˙˙˙˙˙
8
coherence_score%"
coherence_score˙˙˙˙˙˙˙˙˙
*
cohesion
cohesion˙˙˙˙˙˙˙˙˙
6
corrected_text$!
corrected_text˙˙˙˙˙˙˙˙˙
R
dale_chall_readability_score2/
dale_chall_readability_score˙˙˙˙˙˙˙˙˙
B
flesch_kincaid_grade*'
flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
@
flesch_reading_ease)&
flesch_reading_ease˙˙˙˙˙˙˙˙˙
8
freq_diff_words%"
freq_diff_words˙˙˙˙˙˙˙˙˙
0
freq_of_adj!
freq_of_adj˙˙˙˙˙˙˙˙˙
0
freq_of_adv!
freq_of_adv˙˙˙˙˙˙˙˙˙
B
freq_of_distinct_adj*'
freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
B
freq_of_distinct_adv*'
freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
2
freq_of_noun"
freq_of_noun˙˙˙˙˙˙˙˙˙
8
freq_of_pronoun%"
freq_of_pronoun˙˙˙˙˙˙˙˙˙
>
freq_of_transition(%
freq_of_transition˙˙˙˙˙˙˙˙˙
2
freq_of_verb"
freq_of_verb˙˙˙˙˙˙˙˙˙
@
freq_of_wrong_words)&
freq_of_wrong_words˙˙˙˙˙˙˙˙˙
B
lexrank_avg_min_diff*'
lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
D
lexrank_interquartile+(
lexrank_interquartile˙˙˙˙˙˙˙˙˙
6
mcalpine_eflaw$!
mcalpine_eflaw˙˙˙˙˙˙˙˙˙
0
noun_to_adj!
noun_to_adj˙˙˙˙˙˙˙˙˙
D
num_of_grammar_errors+(
num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
>
num_of_short_forms(%
num_of_short_forms˙˙˙˙˙˙˙˙˙	
B
number_of_diff_words*'
number_of_diff_words˙˙˙˙˙˙˙˙˙	
8
number_of_words%"
number_of_words˙˙˙˙˙˙˙˙˙	
:
phrase_diversity&#
phrase_diversity˙˙˙˙˙˙˙˙˙
2
punctuations"
punctuations˙˙˙˙˙˙˙˙˙
@
sentence_complexity)&
sentence_complexity˙˙˙˙˙˙˙˙˙
>
sentiment_compound(%
sentiment_compound˙˙˙˙˙˙˙˙˙
>
sentiment_negative(%
sentiment_negative˙˙˙˙˙˙˙˙˙
>
sentiment_positive(%
sentiment_positive˙˙˙˙˙˙˙˙˙
@
stopwords_frequency)&
stopwords_frequency˙˙˙˙˙˙˙˙˙
4
text_standard# 
text_standard˙˙˙˙˙˙˙˙˙
 
ttr
ttr˙˙˙˙˙˙˙˙˙
0
verb_to_adv!
verb_to_adv˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
,__inference_sequential_2_layer_call_fn_52505ß%&-.ş˘ś
Ž˘Ş
Ş
'
ARI 

inputs/ARI˙˙˙˙˙˙˙˙˙	
I
Incorrect_form_ratio1.
inputs/Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
?
av_word_per_sen,)
inputs/av_word_per_sen˙˙˙˙˙˙˙˙˙
?
coherence_score,)
inputs/coherence_score˙˙˙˙˙˙˙˙˙
1
cohesion%"
inputs/cohesion˙˙˙˙˙˙˙˙˙
=
corrected_text+(
inputs/corrected_text˙˙˙˙˙˙˙˙˙
Y
dale_chall_readability_score96
#inputs/dale_chall_readability_score˙˙˙˙˙˙˙˙˙
I
flesch_kincaid_grade1.
inputs/flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
G
flesch_reading_ease0-
inputs/flesch_reading_ease˙˙˙˙˙˙˙˙˙
?
freq_diff_words,)
inputs/freq_diff_words˙˙˙˙˙˙˙˙˙
7
freq_of_adj(%
inputs/freq_of_adj˙˙˙˙˙˙˙˙˙
7
freq_of_adv(%
inputs/freq_of_adv˙˙˙˙˙˙˙˙˙
I
freq_of_distinct_adj1.
inputs/freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
I
freq_of_distinct_adv1.
inputs/freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
9
freq_of_noun)&
inputs/freq_of_noun˙˙˙˙˙˙˙˙˙
?
freq_of_pronoun,)
inputs/freq_of_pronoun˙˙˙˙˙˙˙˙˙
E
freq_of_transition/,
inputs/freq_of_transition˙˙˙˙˙˙˙˙˙
9
freq_of_verb)&
inputs/freq_of_verb˙˙˙˙˙˙˙˙˙
G
freq_of_wrong_words0-
inputs/freq_of_wrong_words˙˙˙˙˙˙˙˙˙
I
lexrank_avg_min_diff1.
inputs/lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
K
lexrank_interquartile2/
inputs/lexrank_interquartile˙˙˙˙˙˙˙˙˙
=
mcalpine_eflaw+(
inputs/mcalpine_eflaw˙˙˙˙˙˙˙˙˙
7
noun_to_adj(%
inputs/noun_to_adj˙˙˙˙˙˙˙˙˙
K
num_of_grammar_errors2/
inputs/num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
E
num_of_short_forms/,
inputs/num_of_short_forms˙˙˙˙˙˙˙˙˙	
I
number_of_diff_words1.
inputs/number_of_diff_words˙˙˙˙˙˙˙˙˙	
?
number_of_words,)
inputs/number_of_words˙˙˙˙˙˙˙˙˙	
A
phrase_diversity-*
inputs/phrase_diversity˙˙˙˙˙˙˙˙˙
9
punctuations)&
inputs/punctuations˙˙˙˙˙˙˙˙˙
G
sentence_complexity0-
inputs/sentence_complexity˙˙˙˙˙˙˙˙˙
E
sentiment_compound/,
inputs/sentiment_compound˙˙˙˙˙˙˙˙˙
E
sentiment_negative/,
inputs/sentiment_negative˙˙˙˙˙˙˙˙˙
E
sentiment_positive/,
inputs/sentiment_positive˙˙˙˙˙˙˙˙˙
G
stopwords_frequency0-
inputs/stopwords_frequency˙˙˙˙˙˙˙˙˙
;
text_standard*'
inputs/text_standard˙˙˙˙˙˙˙˙˙
'
ttr 

inputs/ttr˙˙˙˙˙˙˙˙˙
7
verb_to_adv(%
inputs/verb_to_adv˙˙˙˙˙˙˙˙˙
p 

 
Ş "˙˙˙˙˙˙˙˙˙
,__inference_sequential_2_layer_call_fn_52558ß%&-.ş˘ś
Ž˘Ş
Ş
'
ARI 

inputs/ARI˙˙˙˙˙˙˙˙˙	
I
Incorrect_form_ratio1.
inputs/Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
?
av_word_per_sen,)
inputs/av_word_per_sen˙˙˙˙˙˙˙˙˙
?
coherence_score,)
inputs/coherence_score˙˙˙˙˙˙˙˙˙
1
cohesion%"
inputs/cohesion˙˙˙˙˙˙˙˙˙
=
corrected_text+(
inputs/corrected_text˙˙˙˙˙˙˙˙˙
Y
dale_chall_readability_score96
#inputs/dale_chall_readability_score˙˙˙˙˙˙˙˙˙
I
flesch_kincaid_grade1.
inputs/flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
G
flesch_reading_ease0-
inputs/flesch_reading_ease˙˙˙˙˙˙˙˙˙
?
freq_diff_words,)
inputs/freq_diff_words˙˙˙˙˙˙˙˙˙
7
freq_of_adj(%
inputs/freq_of_adj˙˙˙˙˙˙˙˙˙
7
freq_of_adv(%
inputs/freq_of_adv˙˙˙˙˙˙˙˙˙
I
freq_of_distinct_adj1.
inputs/freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
I
freq_of_distinct_adv1.
inputs/freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
9
freq_of_noun)&
inputs/freq_of_noun˙˙˙˙˙˙˙˙˙
?
freq_of_pronoun,)
inputs/freq_of_pronoun˙˙˙˙˙˙˙˙˙
E
freq_of_transition/,
inputs/freq_of_transition˙˙˙˙˙˙˙˙˙
9
freq_of_verb)&
inputs/freq_of_verb˙˙˙˙˙˙˙˙˙
G
freq_of_wrong_words0-
inputs/freq_of_wrong_words˙˙˙˙˙˙˙˙˙
I
lexrank_avg_min_diff1.
inputs/lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
K
lexrank_interquartile2/
inputs/lexrank_interquartile˙˙˙˙˙˙˙˙˙
=
mcalpine_eflaw+(
inputs/mcalpine_eflaw˙˙˙˙˙˙˙˙˙
7
noun_to_adj(%
inputs/noun_to_adj˙˙˙˙˙˙˙˙˙
K
num_of_grammar_errors2/
inputs/num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
E
num_of_short_forms/,
inputs/num_of_short_forms˙˙˙˙˙˙˙˙˙	
I
number_of_diff_words1.
inputs/number_of_diff_words˙˙˙˙˙˙˙˙˙	
?
number_of_words,)
inputs/number_of_words˙˙˙˙˙˙˙˙˙	
A
phrase_diversity-*
inputs/phrase_diversity˙˙˙˙˙˙˙˙˙
9
punctuations)&
inputs/punctuations˙˙˙˙˙˙˙˙˙
G
sentence_complexity0-
inputs/sentence_complexity˙˙˙˙˙˙˙˙˙
E
sentiment_compound/,
inputs/sentiment_compound˙˙˙˙˙˙˙˙˙
E
sentiment_negative/,
inputs/sentiment_negative˙˙˙˙˙˙˙˙˙
E
sentiment_positive/,
inputs/sentiment_positive˙˙˙˙˙˙˙˙˙
G
stopwords_frequency0-
inputs/stopwords_frequency˙˙˙˙˙˙˙˙˙
;
text_standard*'
inputs/text_standard˙˙˙˙˙˙˙˙˙
'
ttr 

inputs/ttr˙˙˙˙˙˙˙˙˙
7
verb_to_adv(%
inputs/verb_to_adv˙˙˙˙˙˙˙˙˙
p

 
Ş "˙˙˙˙˙˙˙˙˙
#__inference_signature_wrapper_52452č%&-.¨˘¤
˘ 
Ş
 
ARI
ARI˙˙˙˙˙˙˙˙˙	
B
Incorrect_form_ratio*'
Incorrect_form_ratio˙˙˙˙˙˙˙˙˙
8
av_word_per_sen%"
av_word_per_sen˙˙˙˙˙˙˙˙˙
8
coherence_score%"
coherence_score˙˙˙˙˙˙˙˙˙
*
cohesion
cohesion˙˙˙˙˙˙˙˙˙
6
corrected_text$!
corrected_text˙˙˙˙˙˙˙˙˙
R
dale_chall_readability_score2/
dale_chall_readability_score˙˙˙˙˙˙˙˙˙
B
flesch_kincaid_grade*'
flesch_kincaid_grade˙˙˙˙˙˙˙˙˙
@
flesch_reading_ease)&
flesch_reading_ease˙˙˙˙˙˙˙˙˙
8
freq_diff_words%"
freq_diff_words˙˙˙˙˙˙˙˙˙
0
freq_of_adj!
freq_of_adj˙˙˙˙˙˙˙˙˙
0
freq_of_adv!
freq_of_adv˙˙˙˙˙˙˙˙˙
B
freq_of_distinct_adj*'
freq_of_distinct_adj˙˙˙˙˙˙˙˙˙
B
freq_of_distinct_adv*'
freq_of_distinct_adv˙˙˙˙˙˙˙˙˙
2
freq_of_noun"
freq_of_noun˙˙˙˙˙˙˙˙˙
8
freq_of_pronoun%"
freq_of_pronoun˙˙˙˙˙˙˙˙˙
>
freq_of_transition(%
freq_of_transition˙˙˙˙˙˙˙˙˙
2
freq_of_verb"
freq_of_verb˙˙˙˙˙˙˙˙˙
@
freq_of_wrong_words)&
freq_of_wrong_words˙˙˙˙˙˙˙˙˙
B
lexrank_avg_min_diff*'
lexrank_avg_min_diff˙˙˙˙˙˙˙˙˙
D
lexrank_interquartile+(
lexrank_interquartile˙˙˙˙˙˙˙˙˙
6
mcalpine_eflaw$!
mcalpine_eflaw˙˙˙˙˙˙˙˙˙
0
noun_to_adj!
noun_to_adj˙˙˙˙˙˙˙˙˙
D
num_of_grammar_errors+(
num_of_grammar_errors˙˙˙˙˙˙˙˙˙	
>
num_of_short_forms(%
num_of_short_forms˙˙˙˙˙˙˙˙˙	
B
number_of_diff_words*'
number_of_diff_words˙˙˙˙˙˙˙˙˙	
8
number_of_words%"
number_of_words˙˙˙˙˙˙˙˙˙	
:
phrase_diversity&#
phrase_diversity˙˙˙˙˙˙˙˙˙
2
punctuations"
punctuations˙˙˙˙˙˙˙˙˙
@
sentence_complexity)&
sentence_complexity˙˙˙˙˙˙˙˙˙
>
sentiment_compound(%
sentiment_compound˙˙˙˙˙˙˙˙˙
>
sentiment_negative(%
sentiment_negative˙˙˙˙˙˙˙˙˙
>
sentiment_positive(%
sentiment_positive˙˙˙˙˙˙˙˙˙
@
stopwords_frequency)&
stopwords_frequency˙˙˙˙˙˙˙˙˙
4
text_standard# 
text_standard˙˙˙˙˙˙˙˙˙
 
ttr
ttr˙˙˙˙˙˙˙˙˙
0
verb_to_adv!
verb_to_adv˙˙˙˙˙˙˙˙˙"3Ş0
.
output_1"
output_1˙˙˙˙˙˙˙˙˙