??:
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.9.12v2.9.0-18-gd8ce9f9c3018??0
?
Adam/conv2d_56/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_56/bias/v
{
)Adam/conv2d_56/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_56/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_56/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_56/kernel/v
?
+Adam/conv2d_56/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_56/kernel/v*&
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_53/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_53/beta/v
?
6Adam/batch_normalization_53/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_53/beta/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_53/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_53/gamma/v
?
7Adam/batch_normalization_53/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_53/gamma/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_55/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_55/bias/v
{
)Adam/conv2d_55/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_55/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_55/kernel/v
?
+Adam/conv2d_55/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_11/bias/v
?
3Adam/conv2d_transpose_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_11/bias/v*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_11/kernel/v
?
5Adam/conv2d_transpose_11/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_11/kernel/v*&
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_51/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_51/beta/v
?
6Adam/batch_normalization_51/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_51/beta/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_51/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_51/gamma/v
?
7Adam/batch_normalization_51/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_51/gamma/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_53/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_53/bias/v
{
)Adam/conv2d_53/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_53/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2d_53/kernel/v
?
+Adam/conv2d_53/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/kernel/v*&
_output_shapes
:@ *
dtype0
?
Adam/conv2d_transpose_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_10/bias/v
?
3Adam/conv2d_transpose_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_10/bias/v*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_10/kernel/v
?
5Adam/conv2d_transpose_10/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_10/kernel/v*&
_output_shapes
: @*
dtype0
?
"Adam/batch_normalization_49/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_49/beta/v
?
6Adam/batch_normalization_49/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_49/beta/v*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_49/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_49/gamma/v
?
7Adam/batch_normalization_49/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_49/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_51/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_51/bias/v
{
)Adam/conv2d_51/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_51/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*(
shared_nameAdam/conv2d_51/kernel/v
?
+Adam/conv2d_51/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/kernel/v*'
_output_shapes
:?@*
dtype0
?
Adam/conv2d_transpose_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_9/bias/v
?
2Adam/conv2d_transpose_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_9/bias/v*
_output_shapes
:@*
dtype0
?
 Adam/conv2d_transpose_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*1
shared_name" Adam/conv2d_transpose_9/kernel/v
?
4Adam/conv2d_transpose_9/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_9/kernel/v*'
_output_shapes
:@?*
dtype0
?
"Adam/batch_normalization_47/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_47/beta/v
?
6Adam/batch_normalization_47/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_47/beta/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_47/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_47/gamma/v
?
7Adam/batch_normalization_47/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_47/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_49/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_49/bias/v
|
)Adam/conv2d_49/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_49/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_49/kernel/v
?
+Adam/conv2d_49/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_transpose_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/conv2d_transpose_8/bias/v
?
2Adam/conv2d_transpose_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_8/bias/v*
_output_shapes	
:?*
dtype0
?
 Adam/conv2d_transpose_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*1
shared_name" Adam/conv2d_transpose_8/kernel/v
?
4Adam/conv2d_transpose_8/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_8/kernel/v*(
_output_shapes
:??*
dtype0
?
"Adam/batch_normalization_45/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_45/beta/v
?
6Adam/batch_normalization_45/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_45/beta/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_45/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_45/gamma/v
?
7Adam/batch_normalization_45/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_45/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_47/bias/v
|
)Adam/conv2d_47/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_47/kernel/v
?
+Adam/conv2d_47/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/v*(
_output_shapes
:??*
dtype0
?
"Adam/batch_normalization_43/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_43/beta/v
?
6Adam/batch_normalization_43/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_43/beta/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_43/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_43/gamma/v
?
7Adam/batch_normalization_43/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_43/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_45/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_45/bias/v
|
)Adam/conv2d_45/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_45/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_45/kernel/v
?
+Adam/conv2d_45/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/v*'
_output_shapes
:@?*
dtype0
?
"Adam/batch_normalization_41/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_41/beta/v
?
6Adam/batch_normalization_41/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_41/beta/v*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_41/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_41/gamma/v
?
7Adam/batch_normalization_41/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_41/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_43/bias/v
{
)Adam/conv2d_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_43/kernel/v
?
+Adam/conv2d_43/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/kernel/v*&
_output_shapes
: @*
dtype0
?
"Adam/batch_normalization_39/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_39/beta/v
?
6Adam/batch_normalization_39/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_39/beta/v*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_39/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_39/gamma/v
?
7Adam/batch_normalization_39/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_39/gamma/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_41/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_41/bias/v
{
)Adam/conv2d_41/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_41/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_41/kernel/v
?
+Adam/conv2d_41/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/v*&
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_37/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_37/beta/v
?
6Adam/batch_normalization_37/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_37/beta/v*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_37/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_37/gamma/v
?
7Adam/batch_normalization_37/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_37/gamma/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_39/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_39/bias/v
{
)Adam/conv2d_39/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_39/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_39/kernel/v
?
+Adam/conv2d_39/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_56/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_56/bias/m
{
)Adam/conv2d_56/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_56/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_56/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_56/kernel/m
?
+Adam/conv2d_56/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_56/kernel/m*&
_output_shapes
:*
dtype0
?
"Adam/batch_normalization_53/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_53/beta/m
?
6Adam/batch_normalization_53/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_53/beta/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_53/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_53/gamma/m
?
7Adam/batch_normalization_53/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_53/gamma/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_55/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_55/bias/m
{
)Adam/conv2d_55/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_55/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_55/kernel/m
?
+Adam/conv2d_55/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_55/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_11/bias/m
?
3Adam/conv2d_transpose_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_11/bias/m*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_11/kernel/m
?
5Adam/conv2d_transpose_11/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_11/kernel/m*&
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_51/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_51/beta/m
?
6Adam/batch_normalization_51/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_51/beta/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_51/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_51/gamma/m
?
7Adam/batch_normalization_51/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_51/gamma/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_53/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_53/bias/m
{
)Adam/conv2d_53/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_53/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *(
shared_nameAdam/conv2d_53/kernel/m
?
+Adam/conv2d_53/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_53/kernel/m*&
_output_shapes
:@ *
dtype0
?
Adam/conv2d_transpose_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_10/bias/m
?
3Adam/conv2d_transpose_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_10/bias/m*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*2
shared_name#!Adam/conv2d_transpose_10/kernel/m
?
5Adam/conv2d_transpose_10/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_10/kernel/m*&
_output_shapes
: @*
dtype0
?
"Adam/batch_normalization_49/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_49/beta/m
?
6Adam/batch_normalization_49/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_49/beta/m*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_49/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_49/gamma/m
?
7Adam/batch_normalization_49/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_49/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_51/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_51/bias/m
{
)Adam/conv2d_51/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_51/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*(
shared_nameAdam/conv2d_51/kernel/m
?
+Adam/conv2d_51/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_51/kernel/m*'
_output_shapes
:?@*
dtype0
?
Adam/conv2d_transpose_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/conv2d_transpose_9/bias/m
?
2Adam/conv2d_transpose_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_9/bias/m*
_output_shapes
:@*
dtype0
?
 Adam/conv2d_transpose_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*1
shared_name" Adam/conv2d_transpose_9/kernel/m
?
4Adam/conv2d_transpose_9/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_9/kernel/m*'
_output_shapes
:@?*
dtype0
?
"Adam/batch_normalization_47/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_47/beta/m
?
6Adam/batch_normalization_47/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_47/beta/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_47/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_47/gamma/m
?
7Adam/batch_normalization_47/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_47/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_49/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_49/bias/m
|
)Adam/conv2d_49/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_49/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_49/kernel/m
?
+Adam/conv2d_49/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_49/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_transpose_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*/
shared_name Adam/conv2d_transpose_8/bias/m
?
2Adam/conv2d_transpose_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_8/bias/m*
_output_shapes	
:?*
dtype0
?
 Adam/conv2d_transpose_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*1
shared_name" Adam/conv2d_transpose_8/kernel/m
?
4Adam/conv2d_transpose_8/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_8/kernel/m*(
_output_shapes
:??*
dtype0
?
"Adam/batch_normalization_45/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_45/beta/m
?
6Adam/batch_normalization_45/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_45/beta/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_45/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_45/gamma/m
?
7Adam/batch_normalization_45/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_45/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_47/bias/m
|
)Adam/conv2d_47/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_47/kernel/m
?
+Adam/conv2d_47/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_47/kernel/m*(
_output_shapes
:??*
dtype0
?
"Adam/batch_normalization_43/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_43/beta/m
?
6Adam/batch_normalization_43/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_43/beta/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_43/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_43/gamma/m
?
7Adam/batch_normalization_43/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_43/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_45/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_45/bias/m
|
)Adam/conv2d_45/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_45/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_45/kernel/m
?
+Adam/conv2d_45/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_45/kernel/m*'
_output_shapes
:@?*
dtype0
?
"Adam/batch_normalization_41/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_41/beta/m
?
6Adam/batch_normalization_41/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_41/beta/m*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_41/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_41/gamma/m
?
7Adam/batch_normalization_41/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_41/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_43/bias/m
{
)Adam/conv2d_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_43/kernel/m
?
+Adam/conv2d_43/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_43/kernel/m*&
_output_shapes
: @*
dtype0
?
"Adam/batch_normalization_39/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"Adam/batch_normalization_39/beta/m
?
6Adam/batch_normalization_39/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_39/beta/m*
_output_shapes
: *
dtype0
?
#Adam/batch_normalization_39/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#Adam/batch_normalization_39/gamma/m
?
7Adam/batch_normalization_39/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_39/gamma/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_41/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_41/bias/m
{
)Adam/conv2d_41/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_41/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_41/kernel/m
?
+Adam/conv2d_41/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_41/kernel/m*&
_output_shapes
: *
dtype0
?
"Adam/batch_normalization_37/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_37/beta/m
?
6Adam/batch_normalization_37/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_37/beta/m*
_output_shapes
:*
dtype0
?
#Adam/batch_normalization_37/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_37/gamma/m
?
7Adam/batch_normalization_37/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_37/gamma/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_39/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_39/bias/m
{
)Adam/conv2d_39/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_39/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_39/kernel/m
?
+Adam/conv2d_39/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_39/kernel/m*&
_output_shapes
:*
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
t
conv2d_56/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_56/bias
m
"conv2d_56/bias/Read/ReadVariableOpReadVariableOpconv2d_56/bias*
_output_shapes
:*
dtype0
?
conv2d_56/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_56/kernel
}
$conv2d_56/kernel/Read/ReadVariableOpReadVariableOpconv2d_56/kernel*&
_output_shapes
:*
dtype0
?
&batch_normalization_53/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_53/moving_variance
?
:batch_normalization_53/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_53/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_53/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_53/moving_mean
?
6batch_normalization_53/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_53/moving_mean*
_output_shapes
:*
dtype0
?
batch_normalization_53/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_53/beta
?
/batch_normalization_53/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_53/beta*
_output_shapes
:*
dtype0
?
batch_normalization_53/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_53/gamma
?
0batch_normalization_53/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_53/gamma*
_output_shapes
:*
dtype0
t
conv2d_55/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_55/bias
m
"conv2d_55/bias/Read/ReadVariableOpReadVariableOpconv2d_55/bias*
_output_shapes
:*
dtype0
?
conv2d_55/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_55/kernel
}
$conv2d_55/kernel/Read/ReadVariableOpReadVariableOpconv2d_55/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_11/bias
?
,conv2d_transpose_11/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_11/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_11/kernel
?
.conv2d_transpose_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_11/kernel*&
_output_shapes
: *
dtype0
?
&batch_normalization_51/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_51/moving_variance
?
:batch_normalization_51/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_51/moving_variance*
_output_shapes
: *
dtype0
?
"batch_normalization_51/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_51/moving_mean
?
6batch_normalization_51/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_51/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_51/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_51/beta
?
/batch_normalization_51/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_51/beta*
_output_shapes
: *
dtype0
?
batch_normalization_51/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_51/gamma
?
0batch_normalization_51/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_51/gamma*
_output_shapes
: *
dtype0
t
conv2d_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_53/bias
m
"conv2d_53/bias/Read/ReadVariableOpReadVariableOpconv2d_53/bias*
_output_shapes
: *
dtype0
?
conv2d_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *!
shared_nameconv2d_53/kernel
}
$conv2d_53/kernel/Read/ReadVariableOpReadVariableOpconv2d_53/kernel*&
_output_shapes
:@ *
dtype0
?
conv2d_transpose_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_10/bias
?
,conv2d_transpose_10/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_10/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*+
shared_nameconv2d_transpose_10/kernel
?
.conv2d_transpose_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_10/kernel*&
_output_shapes
: @*
dtype0
?
&batch_normalization_49/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_49/moving_variance
?
:batch_normalization_49/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_49/moving_variance*
_output_shapes
:@*
dtype0
?
"batch_normalization_49/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_49/moving_mean
?
6batch_normalization_49/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_49/moving_mean*
_output_shapes
:@*
dtype0
?
batch_normalization_49/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_49/beta
?
/batch_normalization_49/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_49/beta*
_output_shapes
:@*
dtype0
?
batch_normalization_49/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_49/gamma
?
0batch_normalization_49/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_49/gamma*
_output_shapes
:@*
dtype0
t
conv2d_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_51/bias
m
"conv2d_51/bias/Read/ReadVariableOpReadVariableOpconv2d_51/bias*
_output_shapes
:@*
dtype0
?
conv2d_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*!
shared_nameconv2d_51/kernel
~
$conv2d_51/kernel/Read/ReadVariableOpReadVariableOpconv2d_51/kernel*'
_output_shapes
:?@*
dtype0
?
conv2d_transpose_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameconv2d_transpose_9/bias

+conv2d_transpose_9/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_9/bias*
_output_shapes
:@*
dtype0
?
conv2d_transpose_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?**
shared_nameconv2d_transpose_9/kernel
?
-conv2d_transpose_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_9/kernel*'
_output_shapes
:@?*
dtype0
?
&batch_normalization_47/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_47/moving_variance
?
:batch_normalization_47/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_47/moving_variance*
_output_shapes	
:?*
dtype0
?
"batch_normalization_47/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_47/moving_mean
?
6batch_normalization_47/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_47/moving_mean*
_output_shapes	
:?*
dtype0
?
batch_normalization_47/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_47/beta
?
/batch_normalization_47/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_47/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_47/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_47/gamma
?
0batch_normalization_47/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_47/gamma*
_output_shapes	
:?*
dtype0
u
conv2d_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_49/bias
n
"conv2d_49/bias/Read/ReadVariableOpReadVariableOpconv2d_49/bias*
_output_shapes	
:?*
dtype0
?
conv2d_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_49/kernel

$conv2d_49/kernel/Read/ReadVariableOpReadVariableOpconv2d_49/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_transpose_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameconv2d_transpose_8/bias
?
+conv2d_transpose_8/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_8/bias*
_output_shapes	
:?*
dtype0
?
conv2d_transpose_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??**
shared_nameconv2d_transpose_8/kernel
?
-conv2d_transpose_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_8/kernel*(
_output_shapes
:??*
dtype0
?
&batch_normalization_45/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_45/moving_variance
?
:batch_normalization_45/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_45/moving_variance*
_output_shapes	
:?*
dtype0
?
"batch_normalization_45/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_45/moving_mean
?
6batch_normalization_45/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_45/moving_mean*
_output_shapes	
:?*
dtype0
?
batch_normalization_45/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_45/beta
?
/batch_normalization_45/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_45/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_45/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_45/gamma
?
0batch_normalization_45/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_45/gamma*
_output_shapes	
:?*
dtype0
u
conv2d_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_47/bias
n
"conv2d_47/bias/Read/ReadVariableOpReadVariableOpconv2d_47/bias*
_output_shapes	
:?*
dtype0
?
conv2d_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_47/kernel

$conv2d_47/kernel/Read/ReadVariableOpReadVariableOpconv2d_47/kernel*(
_output_shapes
:??*
dtype0
?
&batch_normalization_43/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_43/moving_variance
?
:batch_normalization_43/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_43/moving_variance*
_output_shapes	
:?*
dtype0
?
"batch_normalization_43/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_43/moving_mean
?
6batch_normalization_43/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_43/moving_mean*
_output_shapes	
:?*
dtype0
?
batch_normalization_43/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_43/beta
?
/batch_normalization_43/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_43/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_43/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_43/gamma
?
0batch_normalization_43/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_43/gamma*
_output_shapes	
:?*
dtype0
u
conv2d_45/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_45/bias
n
"conv2d_45/bias/Read/ReadVariableOpReadVariableOpconv2d_45/bias*
_output_shapes	
:?*
dtype0
?
conv2d_45/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv2d_45/kernel
~
$conv2d_45/kernel/Read/ReadVariableOpReadVariableOpconv2d_45/kernel*'
_output_shapes
:@?*
dtype0
?
&batch_normalization_41/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_41/moving_variance
?
:batch_normalization_41/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_41/moving_variance*
_output_shapes
:@*
dtype0
?
"batch_normalization_41/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_41/moving_mean
?
6batch_normalization_41/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_41/moving_mean*
_output_shapes
:@*
dtype0
?
batch_normalization_41/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_41/beta
?
/batch_normalization_41/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_41/beta*
_output_shapes
:@*
dtype0
?
batch_normalization_41/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_41/gamma
?
0batch_normalization_41/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_41/gamma*
_output_shapes
:@*
dtype0
t
conv2d_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_43/bias
m
"conv2d_43/bias/Read/ReadVariableOpReadVariableOpconv2d_43/bias*
_output_shapes
:@*
dtype0
?
conv2d_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_43/kernel
}
$conv2d_43/kernel/Read/ReadVariableOpReadVariableOpconv2d_43/kernel*&
_output_shapes
: @*
dtype0
?
&batch_normalization_39/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&batch_normalization_39/moving_variance
?
:batch_normalization_39/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_39/moving_variance*
_output_shapes
: *
dtype0
?
"batch_normalization_39/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"batch_normalization_39/moving_mean
?
6batch_normalization_39/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_39/moving_mean*
_output_shapes
: *
dtype0
?
batch_normalization_39/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_39/beta
?
/batch_normalization_39/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_39/beta*
_output_shapes
: *
dtype0
?
batch_normalization_39/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namebatch_normalization_39/gamma
?
0batch_normalization_39/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_39/gamma*
_output_shapes
: *
dtype0
t
conv2d_41/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_41/bias
m
"conv2d_41/bias/Read/ReadVariableOpReadVariableOpconv2d_41/bias*
_output_shapes
: *
dtype0
?
conv2d_41/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_41/kernel
}
$conv2d_41/kernel/Read/ReadVariableOpReadVariableOpconv2d_41/kernel*&
_output_shapes
: *
dtype0
?
&batch_normalization_37/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_37/moving_variance
?
:batch_normalization_37/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_37/moving_variance*
_output_shapes
:*
dtype0
?
"batch_normalization_37/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_37/moving_mean
?
6batch_normalization_37/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_37/moving_mean*
_output_shapes
:*
dtype0
?
batch_normalization_37/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_37/beta
?
/batch_normalization_37/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_37/beta*
_output_shapes
:*
dtype0
?
batch_normalization_37/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_37/gamma
?
0batch_normalization_37/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_37/gamma*
_output_shapes
:*
dtype0
t
conv2d_39/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_39/bias
m
"conv2d_39/bias/Read/ReadVariableOpReadVariableOpconv2d_39/bias*
_output_shapes
:*
dtype0
?
conv2d_39/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_39/kernel
}
$conv2d_39/kernel/Read/ReadVariableOpReadVariableOpconv2d_39/kernel*&
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
layer-19
layer-20
layer_with_weights-8
layer-21
layer_with_weights-9
layer-22
layer-23
layer_with_weights-10
layer-24
layer-25
layer-26
layer_with_weights-11
layer-27
layer_with_weights-12
layer-28
layer-29
layer_with_weights-13
layer-30
 layer-31
!layer-32
"layer_with_weights-14
"layer-33
#layer_with_weights-15
#layer-34
$layer-35
%layer_with_weights-16
%layer-36
&layer-37
'layer-38
(layer_with_weights-17
(layer-39
)layer_with_weights-18
)layer-40
*layer-41
+layer_with_weights-19
+layer-42
,layer-43
-layer-44
.layer_with_weights-20
.layer-45
/layer_with_weights-21
/layer-46
0layer-47
1layer_with_weights-22
1layer-48
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8_default_save_signature
9	optimizer
:
signatures
#;_self_saveable_object_factories*
'
#<_self_saveable_object_factories* 
?
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias
#E_self_saveable_object_factories
 F_jit_compiled_convolution_op*
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
#R_self_saveable_object_factories*
?
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
#Y_self_saveable_object_factories* 
?
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses
#`_self_saveable_object_factories* 
?
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
g_random_generator
#h_self_saveable_object_factories* 
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias
#q_self_saveable_object_factories
 r_jit_compiled_convolution_op*
?
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
yaxis
	zgamma
{beta
|moving_mean
}moving_variance
#~_self_saveable_object_factories*
?
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories* 
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op*
?
C0
D1
N2
O3
P4
Q5
o6
p7
z8
{9
|10
}11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59
?60
?61
?62
?63*
?
C0
D1
N2
O3
o4
p5
z6
{7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
8_default_save_signature
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateCm?Dm?Nm?Om?om?pm?zm?{m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Cv?Dv?Nv?Ov?ov?pv?zv?{v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?*

?serving_default* 
* 
* 

C0
D1*

C0
D1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_39/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_39/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
 
N0
O1
P2
Q3*

N0
O1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_37/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_37/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_37/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_37/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 

o0
p1*

o0
p1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_41/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_41/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
 
z0
{1
|2
}3*

z0
{1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_39/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_39/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_39/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_39/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_43/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_43/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_41/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_41/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_41/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_41/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_45/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_45/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_43/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_43/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_43/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_43/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_47/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_47/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_45/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_45/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_45/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_45/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_8/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_8/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
a[
VARIABLE_VALUEconv2d_49/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_49/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_47/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_47/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_47/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE&batch_normalization_47/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
jd
VARIABLE_VALUEconv2d_transpose_9/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEconv2d_transpose_9/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
a[
VARIABLE_VALUEconv2d_51/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_51/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_49/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_49/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_49/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE&batch_normalization_49/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ke
VARIABLE_VALUEconv2d_transpose_10/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_10/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
a[
VARIABLE_VALUEconv2d_53/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_53/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_51/gamma6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_51/beta5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_51/moving_mean<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE&batch_normalization_51/moving_variance@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
ke
VARIABLE_VALUEconv2d_transpose_11/kernel7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEconv2d_transpose_11/bias5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
a[
VARIABLE_VALUEconv2d_55/kernel7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_55/bias5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
$
?0
?1
?2
?3*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
lf
VARIABLE_VALUEbatch_normalization_53/gamma6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEbatch_normalization_53/beta5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE"batch_normalization_53/moving_mean<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUE&batch_normalization_53/moving_variance@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
a[
VARIABLE_VALUEconv2d_56/kernel7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEconv2d_56/bias5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
?
P0
Q1
|2
}3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17*
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148*

?0
?1*
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

P0
Q1*
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
* 
* 
* 
* 
* 
* 
* 

|0
}1*
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
* 
* 
* 
* 
* 
* 
* 

?0
?1*
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
* 
* 
* 
* 
* 
* 
* 

?0
?1*
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
* 
* 
* 
* 
* 
* 
* 

?0
?1*
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

?0
?1*
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

?0
?1*
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

?0
?1*
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

?0
?1*
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
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
?}
VARIABLE_VALUEAdam/conv2d_39/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_39/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_37/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_37/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_41/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_41/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_39/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_39/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_43/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_43/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_41/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_41/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_45/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_45/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_43/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_43/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_47/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_47/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_45/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_45/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2d_transpose_8/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_8/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_49/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_49/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_47/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_47/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2d_transpose_9/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_9/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_51/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_51/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_49/gamma/mRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_49/beta/mQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_10/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_10/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_53/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_53/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_51/gamma/mRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_51/beta/mQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_11/kernel/mSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_11/bias/mQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_55/kernel/mSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_55/bias/mQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_53/gamma/mRlayer_with_weights-21/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_53/beta/mQlayer_with_weights-21/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_56/kernel/mSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_56/bias/mQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_39/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_39/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_37/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_37/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_41/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_41/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_39/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_39/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_43/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_43/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_41/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_41/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_45/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_45/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_43/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_43/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_47/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_47/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_45/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_45/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2d_transpose_8/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_8/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_49/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_49/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_47/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_47/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE Adam/conv2d_transpose_9/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_9/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_51/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_51/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_49/gamma/vRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_49/beta/vQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_10/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_10/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_53/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_53/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_51/gamma/vRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_51/beta/vQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE!Adam/conv2d_transpose_11/kernel/vSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUEAdam/conv2d_transpose_11/bias/vQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_55/kernel/vSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_55/bias/vQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_53/gamma/vRlayer_with_weights-21/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_53/beta/vQlayer_with_weights-21/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?~
VARIABLE_VALUEAdam/conv2d_56/kernel/vSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?z
VARIABLE_VALUEAdam/conv2d_56/bias/vQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?
serving_default_imgPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_imgconv2d_39/kernelconv2d_39/biasbatch_normalization_37/gammabatch_normalization_37/beta"batch_normalization_37/moving_mean&batch_normalization_37/moving_varianceconv2d_41/kernelconv2d_41/biasbatch_normalization_39/gammabatch_normalization_39/beta"batch_normalization_39/moving_mean&batch_normalization_39/moving_varianceconv2d_43/kernelconv2d_43/biasbatch_normalization_41/gammabatch_normalization_41/beta"batch_normalization_41/moving_mean&batch_normalization_41/moving_varianceconv2d_45/kernelconv2d_45/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_varianceconv2d_47/kernelconv2d_47/biasbatch_normalization_45/gammabatch_normalization_45/beta"batch_normalization_45/moving_mean&batch_normalization_45/moving_varianceconv2d_transpose_8/kernelconv2d_transpose_8/biasconv2d_49/kernelconv2d_49/biasbatch_normalization_47/gammabatch_normalization_47/beta"batch_normalization_47/moving_mean&batch_normalization_47/moving_varianceconv2d_transpose_9/kernelconv2d_transpose_9/biasconv2d_51/kernelconv2d_51/biasbatch_normalization_49/gammabatch_normalization_49/beta"batch_normalization_49/moving_mean&batch_normalization_49/moving_varianceconv2d_transpose_10/kernelconv2d_transpose_10/biasconv2d_53/kernelconv2d_53/biasbatch_normalization_51/gammabatch_normalization_51/beta"batch_normalization_51/moving_mean&batch_normalization_51/moving_varianceconv2d_transpose_11/kernelconv2d_transpose_11/biasconv2d_55/kernelconv2d_55/biasbatch_normalization_53/gammabatch_normalization_53/beta"batch_normalization_53/moving_mean&batch_normalization_53/moving_varianceconv2d_56/kernelconv2d_56/bias*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_83187
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?A
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_39/kernel/Read/ReadVariableOp"conv2d_39/bias/Read/ReadVariableOp0batch_normalization_37/gamma/Read/ReadVariableOp/batch_normalization_37/beta/Read/ReadVariableOp6batch_normalization_37/moving_mean/Read/ReadVariableOp:batch_normalization_37/moving_variance/Read/ReadVariableOp$conv2d_41/kernel/Read/ReadVariableOp"conv2d_41/bias/Read/ReadVariableOp0batch_normalization_39/gamma/Read/ReadVariableOp/batch_normalization_39/beta/Read/ReadVariableOp6batch_normalization_39/moving_mean/Read/ReadVariableOp:batch_normalization_39/moving_variance/Read/ReadVariableOp$conv2d_43/kernel/Read/ReadVariableOp"conv2d_43/bias/Read/ReadVariableOp0batch_normalization_41/gamma/Read/ReadVariableOp/batch_normalization_41/beta/Read/ReadVariableOp6batch_normalization_41/moving_mean/Read/ReadVariableOp:batch_normalization_41/moving_variance/Read/ReadVariableOp$conv2d_45/kernel/Read/ReadVariableOp"conv2d_45/bias/Read/ReadVariableOp0batch_normalization_43/gamma/Read/ReadVariableOp/batch_normalization_43/beta/Read/ReadVariableOp6batch_normalization_43/moving_mean/Read/ReadVariableOp:batch_normalization_43/moving_variance/Read/ReadVariableOp$conv2d_47/kernel/Read/ReadVariableOp"conv2d_47/bias/Read/ReadVariableOp0batch_normalization_45/gamma/Read/ReadVariableOp/batch_normalization_45/beta/Read/ReadVariableOp6batch_normalization_45/moving_mean/Read/ReadVariableOp:batch_normalization_45/moving_variance/Read/ReadVariableOp-conv2d_transpose_8/kernel/Read/ReadVariableOp+conv2d_transpose_8/bias/Read/ReadVariableOp$conv2d_49/kernel/Read/ReadVariableOp"conv2d_49/bias/Read/ReadVariableOp0batch_normalization_47/gamma/Read/ReadVariableOp/batch_normalization_47/beta/Read/ReadVariableOp6batch_normalization_47/moving_mean/Read/ReadVariableOp:batch_normalization_47/moving_variance/Read/ReadVariableOp-conv2d_transpose_9/kernel/Read/ReadVariableOp+conv2d_transpose_9/bias/Read/ReadVariableOp$conv2d_51/kernel/Read/ReadVariableOp"conv2d_51/bias/Read/ReadVariableOp0batch_normalization_49/gamma/Read/ReadVariableOp/batch_normalization_49/beta/Read/ReadVariableOp6batch_normalization_49/moving_mean/Read/ReadVariableOp:batch_normalization_49/moving_variance/Read/ReadVariableOp.conv2d_transpose_10/kernel/Read/ReadVariableOp,conv2d_transpose_10/bias/Read/ReadVariableOp$conv2d_53/kernel/Read/ReadVariableOp"conv2d_53/bias/Read/ReadVariableOp0batch_normalization_51/gamma/Read/ReadVariableOp/batch_normalization_51/beta/Read/ReadVariableOp6batch_normalization_51/moving_mean/Read/ReadVariableOp:batch_normalization_51/moving_variance/Read/ReadVariableOp.conv2d_transpose_11/kernel/Read/ReadVariableOp,conv2d_transpose_11/bias/Read/ReadVariableOp$conv2d_55/kernel/Read/ReadVariableOp"conv2d_55/bias/Read/ReadVariableOp0batch_normalization_53/gamma/Read/ReadVariableOp/batch_normalization_53/beta/Read/ReadVariableOp6batch_normalization_53/moving_mean/Read/ReadVariableOp:batch_normalization_53/moving_variance/Read/ReadVariableOp$conv2d_56/kernel/Read/ReadVariableOp"conv2d_56/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_39/kernel/m/Read/ReadVariableOp)Adam/conv2d_39/bias/m/Read/ReadVariableOp7Adam/batch_normalization_37/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_37/beta/m/Read/ReadVariableOp+Adam/conv2d_41/kernel/m/Read/ReadVariableOp)Adam/conv2d_41/bias/m/Read/ReadVariableOp7Adam/batch_normalization_39/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_39/beta/m/Read/ReadVariableOp+Adam/conv2d_43/kernel/m/Read/ReadVariableOp)Adam/conv2d_43/bias/m/Read/ReadVariableOp7Adam/batch_normalization_41/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_41/beta/m/Read/ReadVariableOp+Adam/conv2d_45/kernel/m/Read/ReadVariableOp)Adam/conv2d_45/bias/m/Read/ReadVariableOp7Adam/batch_normalization_43/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_43/beta/m/Read/ReadVariableOp+Adam/conv2d_47/kernel/m/Read/ReadVariableOp)Adam/conv2d_47/bias/m/Read/ReadVariableOp7Adam/batch_normalization_45/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_45/beta/m/Read/ReadVariableOp4Adam/conv2d_transpose_8/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_8/bias/m/Read/ReadVariableOp+Adam/conv2d_49/kernel/m/Read/ReadVariableOp)Adam/conv2d_49/bias/m/Read/ReadVariableOp7Adam/batch_normalization_47/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_47/beta/m/Read/ReadVariableOp4Adam/conv2d_transpose_9/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_9/bias/m/Read/ReadVariableOp+Adam/conv2d_51/kernel/m/Read/ReadVariableOp)Adam/conv2d_51/bias/m/Read/ReadVariableOp7Adam/batch_normalization_49/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_49/beta/m/Read/ReadVariableOp5Adam/conv2d_transpose_10/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_10/bias/m/Read/ReadVariableOp+Adam/conv2d_53/kernel/m/Read/ReadVariableOp)Adam/conv2d_53/bias/m/Read/ReadVariableOp7Adam/batch_normalization_51/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_51/beta/m/Read/ReadVariableOp5Adam/conv2d_transpose_11/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_11/bias/m/Read/ReadVariableOp+Adam/conv2d_55/kernel/m/Read/ReadVariableOp)Adam/conv2d_55/bias/m/Read/ReadVariableOp7Adam/batch_normalization_53/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_53/beta/m/Read/ReadVariableOp+Adam/conv2d_56/kernel/m/Read/ReadVariableOp)Adam/conv2d_56/bias/m/Read/ReadVariableOp+Adam/conv2d_39/kernel/v/Read/ReadVariableOp)Adam/conv2d_39/bias/v/Read/ReadVariableOp7Adam/batch_normalization_37/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_37/beta/v/Read/ReadVariableOp+Adam/conv2d_41/kernel/v/Read/ReadVariableOp)Adam/conv2d_41/bias/v/Read/ReadVariableOp7Adam/batch_normalization_39/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_39/beta/v/Read/ReadVariableOp+Adam/conv2d_43/kernel/v/Read/ReadVariableOp)Adam/conv2d_43/bias/v/Read/ReadVariableOp7Adam/batch_normalization_41/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_41/beta/v/Read/ReadVariableOp+Adam/conv2d_45/kernel/v/Read/ReadVariableOp)Adam/conv2d_45/bias/v/Read/ReadVariableOp7Adam/batch_normalization_43/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_43/beta/v/Read/ReadVariableOp+Adam/conv2d_47/kernel/v/Read/ReadVariableOp)Adam/conv2d_47/bias/v/Read/ReadVariableOp7Adam/batch_normalization_45/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_45/beta/v/Read/ReadVariableOp4Adam/conv2d_transpose_8/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_8/bias/v/Read/ReadVariableOp+Adam/conv2d_49/kernel/v/Read/ReadVariableOp)Adam/conv2d_49/bias/v/Read/ReadVariableOp7Adam/batch_normalization_47/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_47/beta/v/Read/ReadVariableOp4Adam/conv2d_transpose_9/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_9/bias/v/Read/ReadVariableOp+Adam/conv2d_51/kernel/v/Read/ReadVariableOp)Adam/conv2d_51/bias/v/Read/ReadVariableOp7Adam/batch_normalization_49/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_49/beta/v/Read/ReadVariableOp5Adam/conv2d_transpose_10/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_10/bias/v/Read/ReadVariableOp+Adam/conv2d_53/kernel/v/Read/ReadVariableOp)Adam/conv2d_53/bias/v/Read/ReadVariableOp7Adam/batch_normalization_51/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_51/beta/v/Read/ReadVariableOp5Adam/conv2d_transpose_11/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_11/bias/v/Read/ReadVariableOp+Adam/conv2d_55/kernel/v/Read/ReadVariableOp)Adam/conv2d_55/bias/v/Read/ReadVariableOp7Adam/batch_normalization_53/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_53/beta/v/Read/ReadVariableOp+Adam/conv2d_56/kernel/v/Read/ReadVariableOp)Adam/conv2d_56/bias/v/Read/ReadVariableOpConst*?
Tin?
?2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_85934
?(
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_39/kernelconv2d_39/biasbatch_normalization_37/gammabatch_normalization_37/beta"batch_normalization_37/moving_mean&batch_normalization_37/moving_varianceconv2d_41/kernelconv2d_41/biasbatch_normalization_39/gammabatch_normalization_39/beta"batch_normalization_39/moving_mean&batch_normalization_39/moving_varianceconv2d_43/kernelconv2d_43/biasbatch_normalization_41/gammabatch_normalization_41/beta"batch_normalization_41/moving_mean&batch_normalization_41/moving_varianceconv2d_45/kernelconv2d_45/biasbatch_normalization_43/gammabatch_normalization_43/beta"batch_normalization_43/moving_mean&batch_normalization_43/moving_varianceconv2d_47/kernelconv2d_47/biasbatch_normalization_45/gammabatch_normalization_45/beta"batch_normalization_45/moving_mean&batch_normalization_45/moving_varianceconv2d_transpose_8/kernelconv2d_transpose_8/biasconv2d_49/kernelconv2d_49/biasbatch_normalization_47/gammabatch_normalization_47/beta"batch_normalization_47/moving_mean&batch_normalization_47/moving_varianceconv2d_transpose_9/kernelconv2d_transpose_9/biasconv2d_51/kernelconv2d_51/biasbatch_normalization_49/gammabatch_normalization_49/beta"batch_normalization_49/moving_mean&batch_normalization_49/moving_varianceconv2d_transpose_10/kernelconv2d_transpose_10/biasconv2d_53/kernelconv2d_53/biasbatch_normalization_51/gammabatch_normalization_51/beta"batch_normalization_51/moving_mean&batch_normalization_51/moving_varianceconv2d_transpose_11/kernelconv2d_transpose_11/biasconv2d_55/kernelconv2d_55/biasbatch_normalization_53/gammabatch_normalization_53/beta"batch_normalization_53/moving_mean&batch_normalization_53/moving_varianceconv2d_56/kernelconv2d_56/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_39/kernel/mAdam/conv2d_39/bias/m#Adam/batch_normalization_37/gamma/m"Adam/batch_normalization_37/beta/mAdam/conv2d_41/kernel/mAdam/conv2d_41/bias/m#Adam/batch_normalization_39/gamma/m"Adam/batch_normalization_39/beta/mAdam/conv2d_43/kernel/mAdam/conv2d_43/bias/m#Adam/batch_normalization_41/gamma/m"Adam/batch_normalization_41/beta/mAdam/conv2d_45/kernel/mAdam/conv2d_45/bias/m#Adam/batch_normalization_43/gamma/m"Adam/batch_normalization_43/beta/mAdam/conv2d_47/kernel/mAdam/conv2d_47/bias/m#Adam/batch_normalization_45/gamma/m"Adam/batch_normalization_45/beta/m Adam/conv2d_transpose_8/kernel/mAdam/conv2d_transpose_8/bias/mAdam/conv2d_49/kernel/mAdam/conv2d_49/bias/m#Adam/batch_normalization_47/gamma/m"Adam/batch_normalization_47/beta/m Adam/conv2d_transpose_9/kernel/mAdam/conv2d_transpose_9/bias/mAdam/conv2d_51/kernel/mAdam/conv2d_51/bias/m#Adam/batch_normalization_49/gamma/m"Adam/batch_normalization_49/beta/m!Adam/conv2d_transpose_10/kernel/mAdam/conv2d_transpose_10/bias/mAdam/conv2d_53/kernel/mAdam/conv2d_53/bias/m#Adam/batch_normalization_51/gamma/m"Adam/batch_normalization_51/beta/m!Adam/conv2d_transpose_11/kernel/mAdam/conv2d_transpose_11/bias/mAdam/conv2d_55/kernel/mAdam/conv2d_55/bias/m#Adam/batch_normalization_53/gamma/m"Adam/batch_normalization_53/beta/mAdam/conv2d_56/kernel/mAdam/conv2d_56/bias/mAdam/conv2d_39/kernel/vAdam/conv2d_39/bias/v#Adam/batch_normalization_37/gamma/v"Adam/batch_normalization_37/beta/vAdam/conv2d_41/kernel/vAdam/conv2d_41/bias/v#Adam/batch_normalization_39/gamma/v"Adam/batch_normalization_39/beta/vAdam/conv2d_43/kernel/vAdam/conv2d_43/bias/v#Adam/batch_normalization_41/gamma/v"Adam/batch_normalization_41/beta/vAdam/conv2d_45/kernel/vAdam/conv2d_45/bias/v#Adam/batch_normalization_43/gamma/v"Adam/batch_normalization_43/beta/vAdam/conv2d_47/kernel/vAdam/conv2d_47/bias/v#Adam/batch_normalization_45/gamma/v"Adam/batch_normalization_45/beta/v Adam/conv2d_transpose_8/kernel/vAdam/conv2d_transpose_8/bias/vAdam/conv2d_49/kernel/vAdam/conv2d_49/bias/v#Adam/batch_normalization_47/gamma/v"Adam/batch_normalization_47/beta/v Adam/conv2d_transpose_9/kernel/vAdam/conv2d_transpose_9/bias/vAdam/conv2d_51/kernel/vAdam/conv2d_51/bias/v#Adam/batch_normalization_49/gamma/v"Adam/batch_normalization_49/beta/v!Adam/conv2d_transpose_10/kernel/vAdam/conv2d_transpose_10/bias/vAdam/conv2d_53/kernel/vAdam/conv2d_53/bias/v#Adam/batch_normalization_51/gamma/v"Adam/batch_normalization_51/beta/v!Adam/conv2d_transpose_11/kernel/vAdam/conv2d_transpose_11/bias/vAdam/conv2d_55/kernel/vAdam/conv2d_55/bias/v#Adam/batch_normalization_53/gamma/v"Adam/batch_normalization_53/beta/vAdam/conv2d_56/kernel/vAdam/conv2d_56/bias/v*?
Tin?
?2?*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_86439??)
?
g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_80606

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_45_layer_call_and_return_conditional_losses_81318

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_49_layer_call_fn_84991

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80923?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_84420

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_53_layer_call_fn_85350

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_81170?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
r
H__inference_concatenate_9_layer_call_and_return_conditional_losses_81445

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :~
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????@@?`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????@@?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????@@@:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
I
-__inference_activation_47_layer_call_fn_84872

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_47_layer_call_and_return_conditional_losses_81431i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_8_layer_call_fn_84752
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_8_layer_call_and_return_conditional_losses_81392i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????  ?:?????????  ?:Z V
0
_output_shapes
:?????????  ?
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????  ?
"
_user_specified_name
inputs/1
?
?
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_84182

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

d
E__inference_dropout_19_layer_call_and_return_conditional_losses_81971

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

d
E__inference_dropout_21_layer_call_and_return_conditional_losses_84959

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????@@?C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????@@?*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????@@?x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????@@?r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????@@?b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????@@?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?

?
D__inference_conv2d_45_layer_call_and_return_conditional_losses_84504

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
F
*__inference_dropout_19_layer_call_fn_84591

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_81346i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_80510

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
c
*__inference_dropout_22_layer_call_fn_85115

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_22_layer_call_and_return_conditional_losses_81833y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_53_layer_call_fn_85141

inputs!
unknown:@ 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_53_layer_call_and_return_conditional_losses_81517y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_80479

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
c
*__inference_dropout_17_layer_call_fn_84340

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_82049w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?

d
E__inference_dropout_17_layer_call_and_return_conditional_losses_84357

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@@ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@@ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@@ w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@@ q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@@ a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
c
*__inference_dropout_21_layer_call_fn_84942

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_81879x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_53_layer_call_fn_85337

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_81139?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_85022

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

d
E__inference_dropout_23_layer_call_and_return_conditional_losses_81787

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??n
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:??????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:??????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:??????????? y
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:??????????? s
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:??????????? c
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:??????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_45_layer_call_fn_84658

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80738?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

d
E__inference_dropout_20_layer_call_and_return_conditional_losses_81925

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????  ?C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????  ?*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????  ?x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????  ?r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????  ?b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
t
H__inference_concatenate_9_layer_call_and_return_conditional_losses_84932
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????@@?`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????@@?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????@@@:?????????@@@:Y U
/
_output_shapes
:?????????@@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@@
"
_user_specified_name
inputs/1
?
?
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_84292

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
c
*__inference_dropout_20_layer_call_fn_84769

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_81925x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
I
-__inference_activation_45_layer_call_fn_84699

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_45_layer_call_and_return_conditional_losses_81378i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?M
__inference__traced_save_85934
file_prefix/
+savev2_conv2d_39_kernel_read_readvariableop-
)savev2_conv2d_39_bias_read_readvariableop;
7savev2_batch_normalization_37_gamma_read_readvariableop:
6savev2_batch_normalization_37_beta_read_readvariableopA
=savev2_batch_normalization_37_moving_mean_read_readvariableopE
Asavev2_batch_normalization_37_moving_variance_read_readvariableop/
+savev2_conv2d_41_kernel_read_readvariableop-
)savev2_conv2d_41_bias_read_readvariableop;
7savev2_batch_normalization_39_gamma_read_readvariableop:
6savev2_batch_normalization_39_beta_read_readvariableopA
=savev2_batch_normalization_39_moving_mean_read_readvariableopE
Asavev2_batch_normalization_39_moving_variance_read_readvariableop/
+savev2_conv2d_43_kernel_read_readvariableop-
)savev2_conv2d_43_bias_read_readvariableop;
7savev2_batch_normalization_41_gamma_read_readvariableop:
6savev2_batch_normalization_41_beta_read_readvariableopA
=savev2_batch_normalization_41_moving_mean_read_readvariableopE
Asavev2_batch_normalization_41_moving_variance_read_readvariableop/
+savev2_conv2d_45_kernel_read_readvariableop-
)savev2_conv2d_45_bias_read_readvariableop;
7savev2_batch_normalization_43_gamma_read_readvariableop:
6savev2_batch_normalization_43_beta_read_readvariableopA
=savev2_batch_normalization_43_moving_mean_read_readvariableopE
Asavev2_batch_normalization_43_moving_variance_read_readvariableop/
+savev2_conv2d_47_kernel_read_readvariableop-
)savev2_conv2d_47_bias_read_readvariableop;
7savev2_batch_normalization_45_gamma_read_readvariableop:
6savev2_batch_normalization_45_beta_read_readvariableopA
=savev2_batch_normalization_45_moving_mean_read_readvariableopE
Asavev2_batch_normalization_45_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_8_kernel_read_readvariableop6
2savev2_conv2d_transpose_8_bias_read_readvariableop/
+savev2_conv2d_49_kernel_read_readvariableop-
)savev2_conv2d_49_bias_read_readvariableop;
7savev2_batch_normalization_47_gamma_read_readvariableop:
6savev2_batch_normalization_47_beta_read_readvariableopA
=savev2_batch_normalization_47_moving_mean_read_readvariableopE
Asavev2_batch_normalization_47_moving_variance_read_readvariableop8
4savev2_conv2d_transpose_9_kernel_read_readvariableop6
2savev2_conv2d_transpose_9_bias_read_readvariableop/
+savev2_conv2d_51_kernel_read_readvariableop-
)savev2_conv2d_51_bias_read_readvariableop;
7savev2_batch_normalization_49_gamma_read_readvariableop:
6savev2_batch_normalization_49_beta_read_readvariableopA
=savev2_batch_normalization_49_moving_mean_read_readvariableopE
Asavev2_batch_normalization_49_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_10_kernel_read_readvariableop7
3savev2_conv2d_transpose_10_bias_read_readvariableop/
+savev2_conv2d_53_kernel_read_readvariableop-
)savev2_conv2d_53_bias_read_readvariableop;
7savev2_batch_normalization_51_gamma_read_readvariableop:
6savev2_batch_normalization_51_beta_read_readvariableopA
=savev2_batch_normalization_51_moving_mean_read_readvariableopE
Asavev2_batch_normalization_51_moving_variance_read_readvariableop9
5savev2_conv2d_transpose_11_kernel_read_readvariableop7
3savev2_conv2d_transpose_11_bias_read_readvariableop/
+savev2_conv2d_55_kernel_read_readvariableop-
)savev2_conv2d_55_bias_read_readvariableop;
7savev2_batch_normalization_53_gamma_read_readvariableop:
6savev2_batch_normalization_53_beta_read_readvariableopA
=savev2_batch_normalization_53_moving_mean_read_readvariableopE
Asavev2_batch_normalization_53_moving_variance_read_readvariableop/
+savev2_conv2d_56_kernel_read_readvariableop-
)savev2_conv2d_56_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_39_kernel_m_read_readvariableop4
0savev2_adam_conv2d_39_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_37_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_37_beta_m_read_readvariableop6
2savev2_adam_conv2d_41_kernel_m_read_readvariableop4
0savev2_adam_conv2d_41_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_39_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_39_beta_m_read_readvariableop6
2savev2_adam_conv2d_43_kernel_m_read_readvariableop4
0savev2_adam_conv2d_43_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_41_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_41_beta_m_read_readvariableop6
2savev2_adam_conv2d_45_kernel_m_read_readvariableop4
0savev2_adam_conv2d_45_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_43_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_43_beta_m_read_readvariableop6
2savev2_adam_conv2d_47_kernel_m_read_readvariableop4
0savev2_adam_conv2d_47_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_45_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_45_beta_m_read_readvariableop?
;savev2_adam_conv2d_transpose_8_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_8_bias_m_read_readvariableop6
2savev2_adam_conv2d_49_kernel_m_read_readvariableop4
0savev2_adam_conv2d_49_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_47_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_47_beta_m_read_readvariableop?
;savev2_adam_conv2d_transpose_9_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_9_bias_m_read_readvariableop6
2savev2_adam_conv2d_51_kernel_m_read_readvariableop4
0savev2_adam_conv2d_51_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_49_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_49_beta_m_read_readvariableop@
<savev2_adam_conv2d_transpose_10_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_10_bias_m_read_readvariableop6
2savev2_adam_conv2d_53_kernel_m_read_readvariableop4
0savev2_adam_conv2d_53_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_51_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_51_beta_m_read_readvariableop@
<savev2_adam_conv2d_transpose_11_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_11_bias_m_read_readvariableop6
2savev2_adam_conv2d_55_kernel_m_read_readvariableop4
0savev2_adam_conv2d_55_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_53_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_53_beta_m_read_readvariableop6
2savev2_adam_conv2d_56_kernel_m_read_readvariableop4
0savev2_adam_conv2d_56_bias_m_read_readvariableop6
2savev2_adam_conv2d_39_kernel_v_read_readvariableop4
0savev2_adam_conv2d_39_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_37_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_37_beta_v_read_readvariableop6
2savev2_adam_conv2d_41_kernel_v_read_readvariableop4
0savev2_adam_conv2d_41_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_39_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_39_beta_v_read_readvariableop6
2savev2_adam_conv2d_43_kernel_v_read_readvariableop4
0savev2_adam_conv2d_43_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_41_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_41_beta_v_read_readvariableop6
2savev2_adam_conv2d_45_kernel_v_read_readvariableop4
0savev2_adam_conv2d_45_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_43_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_43_beta_v_read_readvariableop6
2savev2_adam_conv2d_47_kernel_v_read_readvariableop4
0savev2_adam_conv2d_47_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_45_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_45_beta_v_read_readvariableop?
;savev2_adam_conv2d_transpose_8_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_8_bias_v_read_readvariableop6
2savev2_adam_conv2d_49_kernel_v_read_readvariableop4
0savev2_adam_conv2d_49_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_47_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_47_beta_v_read_readvariableop?
;savev2_adam_conv2d_transpose_9_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_9_bias_v_read_readvariableop6
2savev2_adam_conv2d_51_kernel_v_read_readvariableop4
0savev2_adam_conv2d_51_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_49_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_49_beta_v_read_readvariableop@
<savev2_adam_conv2d_transpose_10_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_10_bias_v_read_readvariableop6
2savev2_adam_conv2d_53_kernel_v_read_readvariableop4
0savev2_adam_conv2d_53_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_51_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_51_beta_v_read_readvariableop@
<savev2_adam_conv2d_transpose_11_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_11_bias_v_read_readvariableop6
2savev2_adam_conv2d_55_kernel_v_read_readvariableop4
0savev2_adam_conv2d_55_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_53_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_53_beta_v_read_readvariableop6
2savev2_adam_conv2d_56_kernel_v_read_readvariableop4
0savev2_adam_conv2d_56_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
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
_temp/part?
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
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?]
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?\
value?\B?\?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-21/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-21/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?J
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_39_kernel_read_readvariableop)savev2_conv2d_39_bias_read_readvariableop7savev2_batch_normalization_37_gamma_read_readvariableop6savev2_batch_normalization_37_beta_read_readvariableop=savev2_batch_normalization_37_moving_mean_read_readvariableopAsavev2_batch_normalization_37_moving_variance_read_readvariableop+savev2_conv2d_41_kernel_read_readvariableop)savev2_conv2d_41_bias_read_readvariableop7savev2_batch_normalization_39_gamma_read_readvariableop6savev2_batch_normalization_39_beta_read_readvariableop=savev2_batch_normalization_39_moving_mean_read_readvariableopAsavev2_batch_normalization_39_moving_variance_read_readvariableop+savev2_conv2d_43_kernel_read_readvariableop)savev2_conv2d_43_bias_read_readvariableop7savev2_batch_normalization_41_gamma_read_readvariableop6savev2_batch_normalization_41_beta_read_readvariableop=savev2_batch_normalization_41_moving_mean_read_readvariableopAsavev2_batch_normalization_41_moving_variance_read_readvariableop+savev2_conv2d_45_kernel_read_readvariableop)savev2_conv2d_45_bias_read_readvariableop7savev2_batch_normalization_43_gamma_read_readvariableop6savev2_batch_normalization_43_beta_read_readvariableop=savev2_batch_normalization_43_moving_mean_read_readvariableopAsavev2_batch_normalization_43_moving_variance_read_readvariableop+savev2_conv2d_47_kernel_read_readvariableop)savev2_conv2d_47_bias_read_readvariableop7savev2_batch_normalization_45_gamma_read_readvariableop6savev2_batch_normalization_45_beta_read_readvariableop=savev2_batch_normalization_45_moving_mean_read_readvariableopAsavev2_batch_normalization_45_moving_variance_read_readvariableop4savev2_conv2d_transpose_8_kernel_read_readvariableop2savev2_conv2d_transpose_8_bias_read_readvariableop+savev2_conv2d_49_kernel_read_readvariableop)savev2_conv2d_49_bias_read_readvariableop7savev2_batch_normalization_47_gamma_read_readvariableop6savev2_batch_normalization_47_beta_read_readvariableop=savev2_batch_normalization_47_moving_mean_read_readvariableopAsavev2_batch_normalization_47_moving_variance_read_readvariableop4savev2_conv2d_transpose_9_kernel_read_readvariableop2savev2_conv2d_transpose_9_bias_read_readvariableop+savev2_conv2d_51_kernel_read_readvariableop)savev2_conv2d_51_bias_read_readvariableop7savev2_batch_normalization_49_gamma_read_readvariableop6savev2_batch_normalization_49_beta_read_readvariableop=savev2_batch_normalization_49_moving_mean_read_readvariableopAsavev2_batch_normalization_49_moving_variance_read_readvariableop5savev2_conv2d_transpose_10_kernel_read_readvariableop3savev2_conv2d_transpose_10_bias_read_readvariableop+savev2_conv2d_53_kernel_read_readvariableop)savev2_conv2d_53_bias_read_readvariableop7savev2_batch_normalization_51_gamma_read_readvariableop6savev2_batch_normalization_51_beta_read_readvariableop=savev2_batch_normalization_51_moving_mean_read_readvariableopAsavev2_batch_normalization_51_moving_variance_read_readvariableop5savev2_conv2d_transpose_11_kernel_read_readvariableop3savev2_conv2d_transpose_11_bias_read_readvariableop+savev2_conv2d_55_kernel_read_readvariableop)savev2_conv2d_55_bias_read_readvariableop7savev2_batch_normalization_53_gamma_read_readvariableop6savev2_batch_normalization_53_beta_read_readvariableop=savev2_batch_normalization_53_moving_mean_read_readvariableopAsavev2_batch_normalization_53_moving_variance_read_readvariableop+savev2_conv2d_56_kernel_read_readvariableop)savev2_conv2d_56_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_39_kernel_m_read_readvariableop0savev2_adam_conv2d_39_bias_m_read_readvariableop>savev2_adam_batch_normalization_37_gamma_m_read_readvariableop=savev2_adam_batch_normalization_37_beta_m_read_readvariableop2savev2_adam_conv2d_41_kernel_m_read_readvariableop0savev2_adam_conv2d_41_bias_m_read_readvariableop>savev2_adam_batch_normalization_39_gamma_m_read_readvariableop=savev2_adam_batch_normalization_39_beta_m_read_readvariableop2savev2_adam_conv2d_43_kernel_m_read_readvariableop0savev2_adam_conv2d_43_bias_m_read_readvariableop>savev2_adam_batch_normalization_41_gamma_m_read_readvariableop=savev2_adam_batch_normalization_41_beta_m_read_readvariableop2savev2_adam_conv2d_45_kernel_m_read_readvariableop0savev2_adam_conv2d_45_bias_m_read_readvariableop>savev2_adam_batch_normalization_43_gamma_m_read_readvariableop=savev2_adam_batch_normalization_43_beta_m_read_readvariableop2savev2_adam_conv2d_47_kernel_m_read_readvariableop0savev2_adam_conv2d_47_bias_m_read_readvariableop>savev2_adam_batch_normalization_45_gamma_m_read_readvariableop=savev2_adam_batch_normalization_45_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_8_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_8_bias_m_read_readvariableop2savev2_adam_conv2d_49_kernel_m_read_readvariableop0savev2_adam_conv2d_49_bias_m_read_readvariableop>savev2_adam_batch_normalization_47_gamma_m_read_readvariableop=savev2_adam_batch_normalization_47_beta_m_read_readvariableop;savev2_adam_conv2d_transpose_9_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_9_bias_m_read_readvariableop2savev2_adam_conv2d_51_kernel_m_read_readvariableop0savev2_adam_conv2d_51_bias_m_read_readvariableop>savev2_adam_batch_normalization_49_gamma_m_read_readvariableop=savev2_adam_batch_normalization_49_beta_m_read_readvariableop<savev2_adam_conv2d_transpose_10_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_10_bias_m_read_readvariableop2savev2_adam_conv2d_53_kernel_m_read_readvariableop0savev2_adam_conv2d_53_bias_m_read_readvariableop>savev2_adam_batch_normalization_51_gamma_m_read_readvariableop=savev2_adam_batch_normalization_51_beta_m_read_readvariableop<savev2_adam_conv2d_transpose_11_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_11_bias_m_read_readvariableop2savev2_adam_conv2d_55_kernel_m_read_readvariableop0savev2_adam_conv2d_55_bias_m_read_readvariableop>savev2_adam_batch_normalization_53_gamma_m_read_readvariableop=savev2_adam_batch_normalization_53_beta_m_read_readvariableop2savev2_adam_conv2d_56_kernel_m_read_readvariableop0savev2_adam_conv2d_56_bias_m_read_readvariableop2savev2_adam_conv2d_39_kernel_v_read_readvariableop0savev2_adam_conv2d_39_bias_v_read_readvariableop>savev2_adam_batch_normalization_37_gamma_v_read_readvariableop=savev2_adam_batch_normalization_37_beta_v_read_readvariableop2savev2_adam_conv2d_41_kernel_v_read_readvariableop0savev2_adam_conv2d_41_bias_v_read_readvariableop>savev2_adam_batch_normalization_39_gamma_v_read_readvariableop=savev2_adam_batch_normalization_39_beta_v_read_readvariableop2savev2_adam_conv2d_43_kernel_v_read_readvariableop0savev2_adam_conv2d_43_bias_v_read_readvariableop>savev2_adam_batch_normalization_41_gamma_v_read_readvariableop=savev2_adam_batch_normalization_41_beta_v_read_readvariableop2savev2_adam_conv2d_45_kernel_v_read_readvariableop0savev2_adam_conv2d_45_bias_v_read_readvariableop>savev2_adam_batch_normalization_43_gamma_v_read_readvariableop=savev2_adam_batch_normalization_43_beta_v_read_readvariableop2savev2_adam_conv2d_47_kernel_v_read_readvariableop0savev2_adam_conv2d_47_bias_v_read_readvariableop>savev2_adam_batch_normalization_45_gamma_v_read_readvariableop=savev2_adam_batch_normalization_45_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_8_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_8_bias_v_read_readvariableop2savev2_adam_conv2d_49_kernel_v_read_readvariableop0savev2_adam_conv2d_49_bias_v_read_readvariableop>savev2_adam_batch_normalization_47_gamma_v_read_readvariableop=savev2_adam_batch_normalization_47_beta_v_read_readvariableop;savev2_adam_conv2d_transpose_9_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_9_bias_v_read_readvariableop2savev2_adam_conv2d_51_kernel_v_read_readvariableop0savev2_adam_conv2d_51_bias_v_read_readvariableop>savev2_adam_batch_normalization_49_gamma_v_read_readvariableop=savev2_adam_batch_normalization_49_beta_v_read_readvariableop<savev2_adam_conv2d_transpose_10_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_10_bias_v_read_readvariableop2savev2_adam_conv2d_53_kernel_v_read_readvariableop0savev2_adam_conv2d_53_bias_v_read_readvariableop>savev2_adam_batch_normalization_51_gamma_v_read_readvariableop=savev2_adam_batch_normalization_51_beta_v_read_readvariableop<savev2_adam_conv2d_transpose_11_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_11_bias_v_read_readvariableop2savev2_adam_conv2d_55_kernel_v_read_readvariableop0savev2_adam_conv2d_55_bias_v_read_readvariableop>savev2_adam_batch_normalization_53_gamma_v_read_readvariableop=savev2_adam_batch_normalization_53_beta_v_read_readvariableop2savev2_adam_conv2d_56_kernel_v_read_readvariableop0savev2_adam_conv2d_56_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *?
dtypes?
?2?	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: ::::::: : : : : : : @:@:@:@:@:@:@?:?:?:?:?:?:??:?:?:?:?:?:??:?:??:?:?:?:?:?:@?:@:?@:@:@:@:@:@: @: :@ : : : : : : :: :::::::: : : : : : : : : ::::: : : : : @:@:@:@:@?:?:?:?:??:?:?:?:??:?:??:?:?:?:@?:@:?@:@:@:@: @: :@ : : : : :: :::::::::: : : : : @:@:@:@:@?:?:?:?:??:?:?:?:??:?:??:?:?:?:@?:@:?@:@:@:@: @: :@ : : : : :: :::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:! 

_output_shapes	
:?:.!*
(
_output_shapes
:??:!"

_output_shapes	
:?:!#

_output_shapes	
:?:!$

_output_shapes	
:?:!%

_output_shapes	
:?:!&

_output_shapes	
:?:-')
'
_output_shapes
:@?: (

_output_shapes
:@:-))
'
_output_shapes
:?@: *

_output_shapes
:@: +

_output_shapes
:@: ,

_output_shapes
:@: -

_output_shapes
:@: .

_output_shapes
:@:,/(
&
_output_shapes
: @: 0

_output_shapes
: :,1(
&
_output_shapes
:@ : 2

_output_shapes
: : 3

_output_shapes
: : 4

_output_shapes
: : 5

_output_shapes
: : 6

_output_shapes
: :,7(
&
_output_shapes
: : 8

_output_shapes
::,9(
&
_output_shapes
: : :

_output_shapes
:: ;

_output_shapes
:: <

_output_shapes
:: =

_output_shapes
:: >

_output_shapes
::,?(
&
_output_shapes
:: @

_output_shapes
::A

_output_shapes
: :B

_output_shapes
: :C

_output_shapes
: :D

_output_shapes
: :E

_output_shapes
: :F

_output_shapes
: :G

_output_shapes
: :H

_output_shapes
: :I

_output_shapes
: :,J(
&
_output_shapes
:: K

_output_shapes
:: L

_output_shapes
:: M

_output_shapes
::,N(
&
_output_shapes
: : O

_output_shapes
: : P

_output_shapes
: : Q

_output_shapes
: :,R(
&
_output_shapes
: @: S

_output_shapes
:@: T

_output_shapes
:@: U

_output_shapes
:@:-V)
'
_output_shapes
:@?:!W

_output_shapes	
:?:!X

_output_shapes	
:?:!Y

_output_shapes	
:?:.Z*
(
_output_shapes
:??:![

_output_shapes	
:?:!\

_output_shapes	
:?:!]

_output_shapes	
:?:.^*
(
_output_shapes
:??:!_

_output_shapes	
:?:.`*
(
_output_shapes
:??:!a

_output_shapes	
:?:!b

_output_shapes	
:?:!c

_output_shapes	
:?:-d)
'
_output_shapes
:@?: e

_output_shapes
:@:-f)
'
_output_shapes
:?@: g

_output_shapes
:@: h

_output_shapes
:@: i

_output_shapes
:@:,j(
&
_output_shapes
: @: k

_output_shapes
: :,l(
&
_output_shapes
:@ : m

_output_shapes
: : n

_output_shapes
: : o

_output_shapes
: :,p(
&
_output_shapes
: : q

_output_shapes
::,r(
&
_output_shapes
: : s

_output_shapes
:: t

_output_shapes
:: u

_output_shapes
::,v(
&
_output_shapes
:: w

_output_shapes
::,x(
&
_output_shapes
:: y

_output_shapes
:: z

_output_shapes
:: {

_output_shapes
::,|(
&
_output_shapes
: : }

_output_shapes
: : ~

_output_shapes
: : 

_output_shapes
: :-?(
&
_output_shapes
: @:!?

_output_shapes
:@:!?

_output_shapes
:@:!?

_output_shapes
:@:.?)
'
_output_shapes
:@?:"?

_output_shapes	
:?:"?

_output_shapes	
:?:"?

_output_shapes	
:?:/?*
(
_output_shapes
:??:"?

_output_shapes	
:?:"?

_output_shapes	
:?:"?

_output_shapes	
:?:/?*
(
_output_shapes
:??:"?

_output_shapes	
:?:/?*
(
_output_shapes
:??:"?

_output_shapes	
:?:"?

_output_shapes	
:?:"?

_output_shapes	
:?:.?)
'
_output_shapes
:@?:!?

_output_shapes
:@:.?)
'
_output_shapes
:?@:!?

_output_shapes
:@:!?

_output_shapes
:@:!?

_output_shapes
:@:-?(
&
_output_shapes
: @:!?

_output_shapes
: :-?(
&
_output_shapes
:@ :!?

_output_shapes
: :!?

_output_shapes
: :!?

_output_shapes
: :-?(
&
_output_shapes
: :!?

_output_shapes
::-?(
&
_output_shapes
: :!?

_output_shapes
::!?

_output_shapes
::!?

_output_shapes
::-?(
&
_output_shapes
::!?

_output_shapes
::?

_output_shapes
: 
?
d
H__inference_activation_53_layer_call_and_return_conditional_losses_85396

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:???????????d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
Z
.__inference_concatenate_10_layer_call_fn_85098
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_10_layer_call_and_return_conditional_losses_81498j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????? :??????????? :[ W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/1
?

d
E__inference_dropout_17_layer_call_and_return_conditional_losses_82049

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????@@ C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????@@ *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@@ w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@@ q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@@ a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
d
H__inference_activation_51_layer_call_and_return_conditional_losses_85223

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:??????????? d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
d
H__inference_activation_49_layer_call_and_return_conditional_losses_85050

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@@@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
d
H__inference_activation_45_layer_call_and_return_conditional_losses_81378

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:??????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_56_layer_call_and_return_conditional_losses_81603

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:???????????d
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
B__inference_model_2_layer_call_and_return_conditional_losses_83046
img)
conv2d_39_82869:
conv2d_39_82871:*
batch_normalization_37_82874:*
batch_normalization_37_82876:*
batch_normalization_37_82878:*
batch_normalization_37_82880:)
conv2d_41_82886: 
conv2d_41_82888: *
batch_normalization_39_82891: *
batch_normalization_39_82893: *
batch_normalization_39_82895: *
batch_normalization_39_82897: )
conv2d_43_82903: @
conv2d_43_82905:@*
batch_normalization_41_82908:@*
batch_normalization_41_82910:@*
batch_normalization_41_82912:@*
batch_normalization_41_82914:@*
conv2d_45_82920:@?
conv2d_45_82922:	?+
batch_normalization_43_82925:	?+
batch_normalization_43_82927:	?+
batch_normalization_43_82929:	?+
batch_normalization_43_82931:	?+
conv2d_47_82937:??
conv2d_47_82939:	?+
batch_normalization_45_82942:	?+
batch_normalization_45_82944:	?+
batch_normalization_45_82946:	?+
batch_normalization_45_82948:	?4
conv2d_transpose_8_82952:??'
conv2d_transpose_8_82954:	?+
conv2d_49_82959:??
conv2d_49_82961:	?+
batch_normalization_47_82964:	?+
batch_normalization_47_82966:	?+
batch_normalization_47_82968:	?+
batch_normalization_47_82970:	?3
conv2d_transpose_9_82974:@?&
conv2d_transpose_9_82976:@*
conv2d_51_82981:?@
conv2d_51_82983:@*
batch_normalization_49_82986:@*
batch_normalization_49_82988:@*
batch_normalization_49_82990:@*
batch_normalization_49_82992:@3
conv2d_transpose_10_82996: @'
conv2d_transpose_10_82998: )
conv2d_53_83003:@ 
conv2d_53_83005: *
batch_normalization_51_83008: *
batch_normalization_51_83010: *
batch_normalization_51_83012: *
batch_normalization_51_83014: 3
conv2d_transpose_11_83018: '
conv2d_transpose_11_83020:)
conv2d_55_83025: 
conv2d_55_83027:*
batch_normalization_53_83030:*
batch_normalization_53_83032:*
batch_normalization_53_83034:*
batch_normalization_53_83036:)
conv2d_56_83040:
conv2d_56_83042:
identity??.batch_normalization_37/StatefulPartitionedCall?.batch_normalization_39/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?.batch_normalization_53/StatefulPartitionedCall?!conv2d_39/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall?!conv2d_45/StatefulPartitionedCall?!conv2d_47/StatefulPartitionedCall?!conv2d_49/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?!conv2d_53/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall?!conv2d_56/StatefulPartitionedCall?+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?"dropout_17/StatefulPartitionedCall?"dropout_18/StatefulPartitionedCall?"dropout_19/StatefulPartitionedCall?"dropout_20/StatefulPartitionedCall?"dropout_21/StatefulPartitionedCall?"dropout_22/StatefulPartitionedCall?"dropout_23/StatefulPartitionedCall?
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCallimgconv2d_39_82869conv2d_39_82871*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_81198?
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0batch_normalization_37_82874batch_normalization_37_82876batch_normalization_37_82878batch_normalization_37_82880*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_80434?
activation_37/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_37_layer_call_and_return_conditional_losses_81218?
max_pooling2d_8/PartitionedCallPartitionedCall&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_80454?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_82088?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0conv2d_41_82886conv2d_41_82888*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_81238?
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0batch_normalization_39_82891batch_normalization_39_82893batch_normalization_39_82895batch_normalization_39_82897*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_80510?
activation_39/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_39_layer_call_and_return_conditional_losses_81258?
max_pooling2d_9/PartitionedCallPartitionedCall&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_80530?
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_82049?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0conv2d_43_82903conv2d_43_82905*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_81278?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0batch_normalization_41_82908batch_normalization_41_82910batch_normalization_41_82912batch_normalization_41_82914*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_80586?
activation_41/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_81298?
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_80606?
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0#^dropout_17/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_82010?
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0conv2d_45_82920conv2d_45_82922*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_81318?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0batch_normalization_43_82925batch_normalization_43_82927batch_normalization_43_82929batch_normalization_43_82931*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_80662?
activation_43/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_43_layer_call_and_return_conditional_losses_81338?
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_80682?
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_81971?
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0conv2d_47_82937conv2d_47_82939*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_81358?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0batch_normalization_45_82942batch_normalization_45_82944batch_normalization_45_82946batch_normalization_45_82948*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80738?
activation_45/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_45_layer_call_and_return_conditional_losses_81378?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall&activation_45/PartitionedCall:output:0conv2d_transpose_8_82952conv2d_transpose_8_82954*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_80786?
concatenate_8/PartitionedCallPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0&activation_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_8_layer_call_and_return_conditional_losses_81392?
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0#^dropout_19/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_81925?
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0conv2d_49_82959conv2d_49_82961*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_81411?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0batch_normalization_47_82964batch_normalization_47_82966batch_normalization_47_82968batch_normalization_47_82970*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80846?
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_47_layer_call_and_return_conditional_losses_81431?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall&activation_47/PartitionedCall:output:0conv2d_transpose_9_82974conv2d_transpose_9_82976*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_80894?
concatenate_9/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0&activation_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_9_layer_call_and_return_conditional_losses_81445?
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0#^dropout_20/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_81879?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0conv2d_51_82981conv2d_51_82983*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_51_layer_call_and_return_conditional_losses_81464?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0batch_normalization_49_82986batch_normalization_49_82988batch_normalization_49_82990batch_normalization_49_82992*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80954?
activation_49/PartitionedCallPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_49_layer_call_and_return_conditional_losses_81484?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall&activation_49/PartitionedCall:output:0conv2d_transpose_10_82996conv2d_transpose_10_82998*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_81002?
concatenate_10/PartitionedCallPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_10_layer_call_and_return_conditional_losses_81498?
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0#^dropout_21/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_22_layer_call_and_return_conditional_losses_81833?
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0conv2d_53_83003conv2d_53_83005*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_53_layer_call_and_return_conditional_losses_81517?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0batch_normalization_51_83008batch_normalization_51_83010batch_normalization_51_83012batch_normalization_51_83014*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_81062?
activation_51/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_51_layer_call_and_return_conditional_losses_81537?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall&activation_51/PartitionedCall:output:0conv2d_transpose_11_83018conv2d_transpose_11_83020*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_81110?
concatenate_11/PartitionedCallPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_11_layer_call_and_return_conditional_losses_81551?
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_23_layer_call_and_return_conditional_losses_81787?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0conv2d_55_83025conv2d_55_83027*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_55_layer_call_and_return_conditional_losses_81570?
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_53_83030batch_normalization_53_83032batch_normalization_53_83034batch_normalization_53_83036*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_81170?
activation_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_53_layer_call_and_return_conditional_losses_81590?
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall&activation_53/PartitionedCall:output:0conv2d_56_83040conv2d_56_83042*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_56_layer_call_and_return_conditional_losses_81603?
IdentityIdentity*conv2d_56/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????

NoOpNoOp/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall"^conv2d_56/StatefulPartitionedCall,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall:V R
1
_output_shapes
:???????????

_user_specified_nameimg
?
?
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_80586

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
*__inference_dropout_23_layer_call_fn_85288

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_23_layer_call_and_return_conditional_losses_81787y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
'__inference_model_2_layer_call_fn_82686
img!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?%

unknown_37:@?

unknown_38:@%

unknown_39:?@

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@$

unknown_45: @

unknown_46: $

unknown_47:@ 

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: $

unknown_53: 

unknown_54:$

unknown_55: 

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:$

unknown_61:

unknown_62:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallimgunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*P
_read_only_resource_inputs2
0.	
 !"#$'()*+,/01234789:;<?@*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_82422y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
1
_output_shapes
:???????????

_user_specified_nameimg
?
s
I__inference_concatenate_11_layer_call_and_return_conditional_losses_81551

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:??????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::???????????:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:YU
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_10_layer_call_fn_84453

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_80606?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_11_layer_call_fn_84581

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_80682?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_47_layer_call_fn_84818

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80815?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_51_layer_call_and_return_conditional_losses_81537

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:??????????? d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
I
-__inference_activation_37_layer_call_fn_84187

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_37_layer_call_and_return_conditional_losses_81218j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
t
H__inference_concatenate_8_layer_call_and_return_conditional_losses_84759
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????  ?`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????  ?:?????????  ?:Z V
0
_output_shapes
:?????????  ?
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:?????????  ?
"
_user_specified_name
inputs/1
?
F
*__inference_dropout_20_layer_call_fn_84764

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_81399i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
c
E__inference_dropout_22_layer_call_and_return_conditional_losses_85120

inputs

identity_1X
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????@e

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_41_layer_call_and_return_conditional_losses_81298

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@@@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_55_layer_call_fn_85314

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_55_layer_call_and_return_conditional_losses_81570y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
F
*__inference_dropout_18_layer_call_fn_84463

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_81306h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80738

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
? 
B__inference_model_2_layer_call_and_return_conditional_losses_82422

inputs)
conv2d_39_82245:
conv2d_39_82247:*
batch_normalization_37_82250:*
batch_normalization_37_82252:*
batch_normalization_37_82254:*
batch_normalization_37_82256:)
conv2d_41_82262: 
conv2d_41_82264: *
batch_normalization_39_82267: *
batch_normalization_39_82269: *
batch_normalization_39_82271: *
batch_normalization_39_82273: )
conv2d_43_82279: @
conv2d_43_82281:@*
batch_normalization_41_82284:@*
batch_normalization_41_82286:@*
batch_normalization_41_82288:@*
batch_normalization_41_82290:@*
conv2d_45_82296:@?
conv2d_45_82298:	?+
batch_normalization_43_82301:	?+
batch_normalization_43_82303:	?+
batch_normalization_43_82305:	?+
batch_normalization_43_82307:	?+
conv2d_47_82313:??
conv2d_47_82315:	?+
batch_normalization_45_82318:	?+
batch_normalization_45_82320:	?+
batch_normalization_45_82322:	?+
batch_normalization_45_82324:	?4
conv2d_transpose_8_82328:??'
conv2d_transpose_8_82330:	?+
conv2d_49_82335:??
conv2d_49_82337:	?+
batch_normalization_47_82340:	?+
batch_normalization_47_82342:	?+
batch_normalization_47_82344:	?+
batch_normalization_47_82346:	?3
conv2d_transpose_9_82350:@?&
conv2d_transpose_9_82352:@*
conv2d_51_82357:?@
conv2d_51_82359:@*
batch_normalization_49_82362:@*
batch_normalization_49_82364:@*
batch_normalization_49_82366:@*
batch_normalization_49_82368:@3
conv2d_transpose_10_82372: @'
conv2d_transpose_10_82374: )
conv2d_53_82379:@ 
conv2d_53_82381: *
batch_normalization_51_82384: *
batch_normalization_51_82386: *
batch_normalization_51_82388: *
batch_normalization_51_82390: 3
conv2d_transpose_11_82394: '
conv2d_transpose_11_82396:)
conv2d_55_82401: 
conv2d_55_82403:*
batch_normalization_53_82406:*
batch_normalization_53_82408:*
batch_normalization_53_82410:*
batch_normalization_53_82412:)
conv2d_56_82416:
conv2d_56_82418:
identity??.batch_normalization_37/StatefulPartitionedCall?.batch_normalization_39/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?.batch_normalization_53/StatefulPartitionedCall?!conv2d_39/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall?!conv2d_45/StatefulPartitionedCall?!conv2d_47/StatefulPartitionedCall?!conv2d_49/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?!conv2d_53/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall?!conv2d_56/StatefulPartitionedCall?+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?"dropout_16/StatefulPartitionedCall?"dropout_17/StatefulPartitionedCall?"dropout_18/StatefulPartitionedCall?"dropout_19/StatefulPartitionedCall?"dropout_20/StatefulPartitionedCall?"dropout_21/StatefulPartitionedCall?"dropout_22/StatefulPartitionedCall?"dropout_23/StatefulPartitionedCall?
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_39_82245conv2d_39_82247*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_81198?
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0batch_normalization_37_82250batch_normalization_37_82252batch_normalization_37_82254batch_normalization_37_82256*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_80434?
activation_37/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_37_layer_call_and_return_conditional_losses_81218?
max_pooling2d_8/PartitionedCallPartitionedCall&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_80454?
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_82088?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0conv2d_41_82262conv2d_41_82264*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_81238?
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0batch_normalization_39_82267batch_normalization_39_82269batch_normalization_39_82271batch_normalization_39_82273*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_80510?
activation_39/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_39_layer_call_and_return_conditional_losses_81258?
max_pooling2d_9/PartitionedCallPartitionedCall&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_80530?
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_82049?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0conv2d_43_82279conv2d_43_82281*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_81278?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0batch_normalization_41_82284batch_normalization_41_82286batch_normalization_41_82288batch_normalization_41_82290*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_80586?
activation_41/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_81298?
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_80606?
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_10/PartitionedCall:output:0#^dropout_17/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_82010?
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0conv2d_45_82296conv2d_45_82298*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_81318?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0batch_normalization_43_82301batch_normalization_43_82303batch_normalization_43_82305batch_normalization_43_82307*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_80662?
activation_43/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_43_layer_call_and_return_conditional_losses_81338?
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_80682?
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_11/PartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_81971?
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0conv2d_47_82313conv2d_47_82315*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_81358?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0batch_normalization_45_82318batch_normalization_45_82320batch_normalization_45_82322batch_normalization_45_82324*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80738?
activation_45/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_45_layer_call_and_return_conditional_losses_81378?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall&activation_45/PartitionedCall:output:0conv2d_transpose_8_82328conv2d_transpose_8_82330*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_80786?
concatenate_8/PartitionedCallPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0&activation_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_8_layer_call_and_return_conditional_losses_81392?
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall&concatenate_8/PartitionedCall:output:0#^dropout_19/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_81925?
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0conv2d_49_82335conv2d_49_82337*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_81411?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0batch_normalization_47_82340batch_normalization_47_82342batch_normalization_47_82344batch_normalization_47_82346*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80846?
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_47_layer_call_and_return_conditional_losses_81431?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall&activation_47/PartitionedCall:output:0conv2d_transpose_9_82350conv2d_transpose_9_82352*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_80894?
concatenate_9/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0&activation_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_9_layer_call_and_return_conditional_losses_81445?
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall&concatenate_9/PartitionedCall:output:0#^dropout_20/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_81879?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0conv2d_51_82357conv2d_51_82359*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_51_layer_call_and_return_conditional_losses_81464?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0batch_normalization_49_82362batch_normalization_49_82364batch_normalization_49_82366batch_normalization_49_82368*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80954?
activation_49/PartitionedCallPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_49_layer_call_and_return_conditional_losses_81484?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall&activation_49/PartitionedCall:output:0conv2d_transpose_10_82372conv2d_transpose_10_82374*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_81002?
concatenate_10/PartitionedCallPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_10_layer_call_and_return_conditional_losses_81498?
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall'concatenate_10/PartitionedCall:output:0#^dropout_21/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_22_layer_call_and_return_conditional_losses_81833?
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0conv2d_53_82379conv2d_53_82381*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_53_layer_call_and_return_conditional_losses_81517?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0batch_normalization_51_82384batch_normalization_51_82386batch_normalization_51_82388batch_normalization_51_82390*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_81062?
activation_51/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_51_layer_call_and_return_conditional_losses_81537?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall&activation_51/PartitionedCall:output:0conv2d_transpose_11_82394conv2d_transpose_11_82396*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_81110?
concatenate_11/PartitionedCallPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_11_layer_call_and_return_conditional_losses_81551?
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall'concatenate_11/PartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_23_layer_call_and_return_conditional_losses_81787?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0conv2d_55_82401conv2d_55_82403*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_55_layer_call_and_return_conditional_losses_81570?
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_53_82406batch_normalization_53_82408batch_normalization_53_82410batch_normalization_53_82412*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_81170?
activation_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_53_layer_call_and_return_conditional_losses_81590?
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall&activation_53/PartitionedCall:output:0conv2d_56_82416conv2d_56_82418*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_56_layer_call_and_return_conditional_losses_81603?
IdentityIdentity*conv2d_56/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????

NoOpNoOp/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall"^conv2d_56/StatefulPartitionedCall,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_81170

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_18_layer_call_and_return_conditional_losses_84473

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????  @c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????  @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?

?
D__inference_conv2d_49_layer_call_and_return_conditional_losses_81411

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_84548

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_16_layer_call_fn_84212

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_82088y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

d
E__inference_dropout_23_layer_call_and_return_conditional_losses_85305

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??n
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:??????????? C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:??????????? *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:??????????? y
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:??????????? s
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:??????????? c
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:??????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
d
H__inference_activation_43_layer_call_and_return_conditional_losses_84576

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????  ?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
? 
?
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_80894

inputsC
(conv2d_transpose_readvariableop_resource:@?-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_81139

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_47_layer_call_fn_84831

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80846?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_43_layer_call_fn_84517

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_80631?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_85368

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_41_layer_call_fn_84389

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_80555?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

d
E__inference_dropout_16_layer_call_and_return_conditional_losses_82088

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??n
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????y
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????s
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????c
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_19_layer_call_and_return_conditional_losses_81346

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

d
E__inference_dropout_19_layer_call_and_return_conditional_losses_84613

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:??????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:??????????x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:??????????r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:??????????b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_81002

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_83187
img!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?%

unknown_37:@?

unknown_38:@%

unknown_39:?@

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@$

unknown_45: @

unknown_46: $

unknown_47:@ 

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: $

unknown_53: 

unknown_54:$

unknown_55: 

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:$

unknown_61:

unknown_62:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallimgunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_80381y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
1
_output_shapes
:???????????

_user_specified_nameimg
?
f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_80454

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_84458

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_23_layer_call_and_return_conditional_losses_85293

inputs

identity_1X
IdentityIdentityinputs*
T0*1
_output_shapes
:??????????? e

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:??????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
??
??
B__inference_model_2_layer_call_and_return_conditional_losses_84101

inputsB
(conv2d_39_conv2d_readvariableop_resource:7
)conv2d_39_biasadd_readvariableop_resource:<
.batch_normalization_37_readvariableop_resource:>
0batch_normalization_37_readvariableop_1_resource:M
?batch_normalization_37_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_41_conv2d_readvariableop_resource: 7
)conv2d_41_biasadd_readvariableop_resource: <
.batch_normalization_39_readvariableop_resource: >
0batch_normalization_39_readvariableop_1_resource: M
?batch_normalization_39_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_43_conv2d_readvariableop_resource: @7
)conv2d_43_biasadd_readvariableop_resource:@<
.batch_normalization_41_readvariableop_resource:@>
0batch_normalization_41_readvariableop_1_resource:@M
?batch_normalization_41_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_45_conv2d_readvariableop_resource:@?8
)conv2d_45_biasadd_readvariableop_resource:	?=
.batch_normalization_43_readvariableop_resource:	??
0batch_normalization_43_readvariableop_1_resource:	?N
?batch_normalization_43_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_47_conv2d_readvariableop_resource:??8
)conv2d_47_biasadd_readvariableop_resource:	?=
.batch_normalization_45_readvariableop_resource:	??
0batch_normalization_45_readvariableop_1_resource:	?N
?batch_normalization_45_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource:	?W
;conv2d_transpose_8_conv2d_transpose_readvariableop_resource:??A
2conv2d_transpose_8_biasadd_readvariableop_resource:	?D
(conv2d_49_conv2d_readvariableop_resource:??8
)conv2d_49_biasadd_readvariableop_resource:	?=
.batch_normalization_47_readvariableop_resource:	??
0batch_normalization_47_readvariableop_1_resource:	?N
?batch_normalization_47_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:	?V
;conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@?@
2conv2d_transpose_9_biasadd_readvariableop_resource:@C
(conv2d_51_conv2d_readvariableop_resource:?@7
)conv2d_51_biasadd_readvariableop_resource:@<
.batch_normalization_49_readvariableop_resource:@>
0batch_normalization_49_readvariableop_1_resource:@M
?batch_normalization_49_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_10_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_10_biasadd_readvariableop_resource: B
(conv2d_53_conv2d_readvariableop_resource:@ 7
)conv2d_53_biasadd_readvariableop_resource: <
.batch_normalization_51_readvariableop_resource: >
0batch_normalization_51_readvariableop_1_resource: M
?batch_normalization_51_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_11_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_11_biasadd_readvariableop_resource:B
(conv2d_55_conv2d_readvariableop_resource: 7
)conv2d_55_biasadd_readvariableop_resource:<
.batch_normalization_53_readvariableop_resource:>
0batch_normalization_53_readvariableop_1_resource:M
?batch_normalization_53_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_56_conv2d_readvariableop_resource:7
)conv2d_56_biasadd_readvariableop_resource:
identity??%batch_normalization_37/AssignNewValue?'batch_normalization_37/AssignNewValue_1?6batch_normalization_37/FusedBatchNormV3/ReadVariableOp?8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_37/ReadVariableOp?'batch_normalization_37/ReadVariableOp_1?%batch_normalization_39/AssignNewValue?'batch_normalization_39/AssignNewValue_1?6batch_normalization_39/FusedBatchNormV3/ReadVariableOp?8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_39/ReadVariableOp?'batch_normalization_39/ReadVariableOp_1?%batch_normalization_41/AssignNewValue?'batch_normalization_41/AssignNewValue_1?6batch_normalization_41/FusedBatchNormV3/ReadVariableOp?8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_41/ReadVariableOp?'batch_normalization_41/ReadVariableOp_1?%batch_normalization_43/AssignNewValue?'batch_normalization_43/AssignNewValue_1?6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_43/ReadVariableOp?'batch_normalization_43/ReadVariableOp_1?%batch_normalization_45/AssignNewValue?'batch_normalization_45/AssignNewValue_1?6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_45/ReadVariableOp?'batch_normalization_45/ReadVariableOp_1?%batch_normalization_47/AssignNewValue?'batch_normalization_47/AssignNewValue_1?6batch_normalization_47/FusedBatchNormV3/ReadVariableOp?8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_47/ReadVariableOp?'batch_normalization_47/ReadVariableOp_1?%batch_normalization_49/AssignNewValue?'batch_normalization_49/AssignNewValue_1?6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_49/ReadVariableOp?'batch_normalization_49/ReadVariableOp_1?%batch_normalization_51/AssignNewValue?'batch_normalization_51/AssignNewValue_1?6batch_normalization_51/FusedBatchNormV3/ReadVariableOp?8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_51/ReadVariableOp?'batch_normalization_51/ReadVariableOp_1?%batch_normalization_53/AssignNewValue?'batch_normalization_53/AssignNewValue_1?6batch_normalization_53/FusedBatchNormV3/ReadVariableOp?8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_53/ReadVariableOp?'batch_normalization_53/ReadVariableOp_1? conv2d_39/BiasAdd/ReadVariableOp?conv2d_39/Conv2D/ReadVariableOp? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp? conv2d_43/BiasAdd/ReadVariableOp?conv2d_43/Conv2D/ReadVariableOp? conv2d_45/BiasAdd/ReadVariableOp?conv2d_45/Conv2D/ReadVariableOp? conv2d_47/BiasAdd/ReadVariableOp?conv2d_47/Conv2D/ReadVariableOp? conv2d_49/BiasAdd/ReadVariableOp?conv2d_49/Conv2D/ReadVariableOp? conv2d_51/BiasAdd/ReadVariableOp?conv2d_51/Conv2D/ReadVariableOp? conv2d_53/BiasAdd/ReadVariableOp?conv2d_53/Conv2D/ReadVariableOp? conv2d_55/BiasAdd/ReadVariableOp?conv2d_55/Conv2D/ReadVariableOp? conv2d_56/BiasAdd/ReadVariableOp?conv2d_56/Conv2D/ReadVariableOp?*conv2d_transpose_10/BiasAdd/ReadVariableOp?3conv2d_transpose_10/conv2d_transpose/ReadVariableOp?*conv2d_transpose_11/BiasAdd/ReadVariableOp?3conv2d_transpose_11/conv2d_transpose/ReadVariableOp?)conv2d_transpose_8/BiasAdd/ReadVariableOp?2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?)conv2d_transpose_9/BiasAdd/ReadVariableOp?2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_39/Conv2DConv2Dinputs'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_37/ReadVariableOpReadVariableOp.batch_normalization_37_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_37/ReadVariableOp_1ReadVariableOp0batch_normalization_37_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_37/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_37_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_37/FusedBatchNormV3FusedBatchNormV3conv2d_39/BiasAdd:output:0-batch_normalization_37/ReadVariableOp:value:0/batch_normalization_37/ReadVariableOp_1:value:0>batch_normalization_37/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_37/AssignNewValueAssignVariableOp?batch_normalization_37_fusedbatchnormv3_readvariableop_resource4batch_normalization_37/FusedBatchNormV3:batch_mean:07^batch_normalization_37/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_37/AssignNewValue_1AssignVariableOpAbatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_37/FusedBatchNormV3:batch_variance:09^batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_37/ReluRelu+batch_normalization_37/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:????????????
max_pooling2d_8/MaxPoolMaxPool activation_37/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
]
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_16/dropout/MulMul max_pooling2d_8/MaxPool:output:0!dropout_16/dropout/Const:output:0*
T0*1
_output_shapes
:???????????h
dropout_16/dropout/ShapeShape max_pooling2d_8/MaxPool:output:0*
T0*
_output_shapes
:?
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype0f
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:????????????
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:????????????
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*1
_output_shapes
:????????????
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_41/Conv2DConv2Ddropout_16/dropout/Mul_1:z:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? ?
%batch_normalization_39/ReadVariableOpReadVariableOp.batch_normalization_39_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_39/ReadVariableOp_1ReadVariableOp0batch_normalization_39_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_39/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_39_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_39/FusedBatchNormV3FusedBatchNormV3conv2d_41/BiasAdd:output:0-batch_normalization_39/ReadVariableOp:value:0/batch_normalization_39/ReadVariableOp_1:value:0>batch_normalization_39/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_39/AssignNewValueAssignVariableOp?batch_normalization_39_fusedbatchnormv3_readvariableop_resource4batch_normalization_39/FusedBatchNormV3:batch_mean:07^batch_normalization_39/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_39/AssignNewValue_1AssignVariableOpAbatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_39/FusedBatchNormV3:batch_variance:09^batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_39/ReluRelu+batch_normalization_39/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? ?
max_pooling2d_9/MaxPoolMaxPool activation_39/Relu:activations:0*/
_output_shapes
:?????????@@ *
ksize
*
paddingVALID*
strides
]
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_17/dropout/MulMul max_pooling2d_9/MaxPool:output:0!dropout_17/dropout/Const:output:0*
T0*/
_output_shapes
:?????????@@ h
dropout_17/dropout/ShapeShape max_pooling2d_9/MaxPool:output:0*
T0*
_output_shapes
:?
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????@@ *
dtype0f
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????@@ ?
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????@@ ?
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????@@ ?
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_43/Conv2DConv2Ddropout_17/dropout/Mul_1:z:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
%batch_normalization_41/ReadVariableOpReadVariableOp.batch_normalization_41_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_41/ReadVariableOp_1ReadVariableOp0batch_normalization_41_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_41/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_41_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_41/FusedBatchNormV3FusedBatchNormV3conv2d_43/BiasAdd:output:0-batch_normalization_41/ReadVariableOp:value:0/batch_normalization_41/ReadVariableOp_1:value:0>batch_normalization_41/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_41/AssignNewValueAssignVariableOp?batch_normalization_41_fusedbatchnormv3_readvariableop_resource4batch_normalization_41/FusedBatchNormV3:batch_mean:07^batch_normalization_41/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_41/AssignNewValue_1AssignVariableOpAbatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_41/FusedBatchNormV3:batch_variance:09^batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_41/ReluRelu+batch_normalization_41/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@@?
max_pooling2d_10/MaxPoolMaxPool activation_41/Relu:activations:0*/
_output_shapes
:?????????  @*
ksize
*
paddingVALID*
strides
]
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_18/dropout/MulMul!max_pooling2d_10/MaxPool:output:0!dropout_18/dropout/Const:output:0*
T0*/
_output_shapes
:?????????  @i
dropout_18/dropout/ShapeShape!max_pooling2d_10/MaxPool:output:0*
T0*
_output_shapes
:?
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????  @*
dtype0f
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  @?
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  @?
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  @?
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_45/Conv2DConv2Ddropout_18/dropout/Mul_1:z:0'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
%batch_normalization_43/ReadVariableOpReadVariableOp.batch_normalization_43_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_43/ReadVariableOp_1ReadVariableOp0batch_normalization_43_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_43/FusedBatchNormV3FusedBatchNormV3conv2d_45/BiasAdd:output:0-batch_normalization_43/ReadVariableOp:value:0/batch_normalization_43/ReadVariableOp_1:value:0>batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_43/AssignNewValueAssignVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource4batch_normalization_43/FusedBatchNormV3:batch_mean:07^batch_normalization_43/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_43/AssignNewValue_1AssignVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_43/FusedBatchNormV3:batch_variance:09^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_43/ReluRelu+batch_normalization_43/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ??
max_pooling2d_11/MaxPoolMaxPool activation_43/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
]
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_19/dropout/MulMul!max_pooling2d_11/MaxPool:output:0!dropout_19/dropout/Const:output:0*
T0*0
_output_shapes
:??????????i
dropout_19/dropout/ShapeShape!max_pooling2d_11/MaxPool:output:0*
T0*
_output_shapes
:?
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*0
_output_shapes
:??????????*
dtype0f
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:???????????
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:???????????
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*0
_output_shapes
:???????????
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_47/Conv2DConv2Ddropout_19/dropout/Mul_1:z:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
%batch_normalization_45/ReadVariableOpReadVariableOp.batch_normalization_45_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_45/ReadVariableOp_1ReadVariableOp0batch_normalization_45_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_45/FusedBatchNormV3FusedBatchNormV3conv2d_47/BiasAdd:output:0-batch_normalization_45/ReadVariableOp:value:0/batch_normalization_45/ReadVariableOp_1:value:0>batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_45/AssignNewValueAssignVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource4batch_normalization_45/FusedBatchNormV3:batch_mean:07^batch_normalization_45/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_45/AssignNewValue_1AssignVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_45/FusedBatchNormV3:batch_variance:09^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_45/ReluRelu+batch_normalization_45/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????h
conv2d_transpose_8/ShapeShape activation_45/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_8/strided_sliceStridedSlice!conv2d_transpose_8/Shape:output:0/conv2d_transpose_8/strided_slice/stack:output:01conv2d_transpose_8/strided_slice/stack_1:output:01conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B : \
conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
conv2d_transpose_8/stackPack)conv2d_transpose_8/strided_slice:output:0#conv2d_transpose_8/stack/1:output:0#conv2d_transpose_8/stack/2:output:0#conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_8/strided_slice_1StridedSlice!conv2d_transpose_8/stack:output:01conv2d_transpose_8/strided_slice_1/stack:output:03conv2d_transpose_8/strided_slice_1/stack_1:output:03conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
#conv2d_transpose_8/conv2d_transposeConv2DBackpropInput!conv2d_transpose_8/stack:output:0:conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0 activation_45/Relu:activations:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
)conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_transpose_8/BiasAddBiasAdd,conv2d_transpose_8/conv2d_transpose:output:01conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?[
concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_8/concatConcatV2#conv2d_transpose_8/BiasAdd:output:0 activation_43/Relu:activations:0"concatenate_8/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????  ?]
dropout_20/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_20/dropout/MulMulconcatenate_8/concat:output:0!dropout_20/dropout/Const:output:0*
T0*0
_output_shapes
:?????????  ?e
dropout_20/dropout/ShapeShapeconcatenate_8/concat:output:0*
T0*
_output_shapes
:?
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????  ?*
dtype0f
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????  ??
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????  ??
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????  ??
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_49/Conv2DConv2Ddropout_20/dropout/Mul_1:z:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3conv2d_49/BiasAdd:output:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_47/AssignNewValueAssignVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource4batch_normalization_47/FusedBatchNormV3:batch_mean:07^batch_normalization_47/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_47/AssignNewValue_1AssignVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_47/FusedBatchNormV3:batch_variance:09^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_47/ReluRelu+batch_normalization_47/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?h
conv2d_transpose_9/ShapeShape activation_47/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_9/strided_sliceStridedSlice!conv2d_transpose_9/Shape:output:0/conv2d_transpose_9/strided_slice/stack:output:01conv2d_transpose_9/strided_slice/stack_1:output:01conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_9/stackPack)conv2d_transpose_9/strided_slice:output:0#conv2d_transpose_9/stack/1:output:0#conv2d_transpose_9/stack/2:output:0#conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_9/strided_slice_1StridedSlice!conv2d_transpose_9/stack:output:01conv2d_transpose_9/strided_slice_1/stack:output:03conv2d_transpose_9/strided_slice_1/stack_1:output:03conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
#conv2d_transpose_9/conv2d_transposeConv2DBackpropInput!conv2d_transpose_9/stack:output:0:conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0 activation_47/Relu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
)conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_9/BiasAddBiasAdd,conv2d_transpose_9/conv2d_transpose:output:01conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@[
concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_9/concatConcatV2#conv2d_transpose_9/BiasAdd:output:0 activation_41/Relu:activations:0"concatenate_9/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????@@?]
dropout_21/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_21/dropout/MulMulconcatenate_9/concat:output:0!dropout_21/dropout/Const:output:0*
T0*0
_output_shapes
:?????????@@?e
dropout_21/dropout/ShapeShapeconcatenate_9/concat:output:0*
T0*
_output_shapes
:?
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????@@?*
dtype0f
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????@@??
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????@@??
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????@@??
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
conv2d_51/Conv2DConv2Ddropout_21/dropout/Mul_1:z:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3conv2d_51/BiasAdd:output:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_49/AssignNewValueAssignVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource4batch_normalization_49/FusedBatchNormV3:batch_mean:07^batch_normalization_49/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_49/AssignNewValue_1AssignVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_49/FusedBatchNormV3:batch_variance:09^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_49/ReluRelu+batch_normalization_49/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@@i
conv2d_transpose_10/ShapeShape activation_49/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_10/strided_sliceStridedSlice"conv2d_transpose_10/Shape:output:00conv2d_transpose_10/strided_slice/stack:output:02conv2d_transpose_10/strided_slice/stack_1:output:02conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?^
conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?]
conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_10/stackPack*conv2d_transpose_10/strided_slice:output:0$conv2d_transpose_10/stack/1:output:0$conv2d_transpose_10/stack/2:output:0$conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_10/strided_slice_1StridedSlice"conv2d_transpose_10/stack:output:02conv2d_transpose_10/strided_slice_1/stack:output:04conv2d_transpose_10/strided_slice_1/stack_1:output:04conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_10_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
$conv2d_transpose_10/conv2d_transposeConv2DBackpropInput"conv2d_transpose_10/stack:output:0;conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0 activation_49/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
*conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_10/BiasAddBiasAdd-conv2d_transpose_10/conv2d_transpose:output:02conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? \
concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_10/concatConcatV2$conv2d_transpose_10/BiasAdd:output:0 activation_39/Relu:activations:0#concatenate_10/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@]
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_22/dropout/MulMulconcatenate_10/concat:output:0!dropout_22/dropout/Const:output:0*
T0*1
_output_shapes
:???????????@f
dropout_22/dropout/ShapeShapeconcatenate_10/concat:output:0*
T0*
_output_shapes
:?
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype0f
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@?
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@?
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@?
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_53/Conv2DConv2Ddropout_22/dropout/Mul_1:z:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? ?
%batch_normalization_51/ReadVariableOpReadVariableOp.batch_normalization_51_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_51/ReadVariableOp_1ReadVariableOp0batch_normalization_51_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_51/FusedBatchNormV3FusedBatchNormV3conv2d_53/BiasAdd:output:0-batch_normalization_51/ReadVariableOp:value:0/batch_normalization_51/ReadVariableOp_1:value:0>batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_51/AssignNewValueAssignVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource4batch_normalization_51/FusedBatchNormV3:batch_mean:07^batch_normalization_51/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_51/AssignNewValue_1AssignVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_51/FusedBatchNormV3:batch_variance:09^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_51/ReluRelu+batch_normalization_51/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? i
conv2d_transpose_11/ShapeShape activation_51/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_11/strided_sliceStridedSlice"conv2d_transpose_11/Shape:output:00conv2d_transpose_11/strided_slice/stack:output:02conv2d_transpose_11/strided_slice/stack_1:output:02conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?^
conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?]
conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_11/stackPack*conv2d_transpose_11/strided_slice:output:0$conv2d_transpose_11/stack/1:output:0$conv2d_transpose_11/stack/2:output:0$conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_11/strided_slice_1StridedSlice"conv2d_transpose_11/stack:output:02conv2d_transpose_11/strided_slice_1/stack:output:04conv2d_transpose_11/strided_slice_1/stack_1:output:04conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_11_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_11/conv2d_transposeConv2DBackpropInput"conv2d_transpose_11/stack:output:0;conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0 activation_51/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
*conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_11/BiasAddBiasAdd-conv2d_transpose_11/conv2d_transpose:output:02conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????\
concatenate_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_11/concatConcatV2$conv2d_transpose_11/BiasAdd:output:0 activation_37/Relu:activations:0#concatenate_11/concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? ]
dropout_23/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8???
dropout_23/dropout/MulMulconcatenate_11/concat:output:0!dropout_23/dropout/Const:output:0*
T0*1
_output_shapes
:??????????? f
dropout_23/dropout/ShapeShapeconcatenate_11/concat:output:0*
T0*
_output_shapes
:?
/dropout_23/dropout/random_uniform/RandomUniformRandomUniform!dropout_23/dropout/Shape:output:0*
T0*1
_output_shapes
:??????????? *
dtype0f
!dropout_23/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout_23/dropout/GreaterEqualGreaterEqual8dropout_23/dropout/random_uniform/RandomUniform:output:0*dropout_23/dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:??????????? ?
dropout_23/dropout/CastCast#dropout_23/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:??????????? ?
dropout_23/dropout/Mul_1Muldropout_23/dropout/Mul:z:0dropout_23/dropout/Cast:y:0*
T0*1
_output_shapes
:??????????? ?
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_55/Conv2DConv2Ddropout_23/dropout/Mul_1:z:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_53/ReadVariableOpReadVariableOp.batch_normalization_53_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_53/ReadVariableOp_1ReadVariableOp0batch_normalization_53_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_53/FusedBatchNormV3FusedBatchNormV3conv2d_55/BiasAdd:output:0-batch_normalization_53/ReadVariableOp:value:0/batch_normalization_53/ReadVariableOp_1:value:0>batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_53/AssignNewValueAssignVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource4batch_normalization_53/FusedBatchNormV3:batch_mean:07^batch_normalization_53/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_53/AssignNewValue_1AssignVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_53/FusedBatchNormV3:batch_variance:09^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
activation_53/ReluRelu+batch_normalization_53/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:????????????
conv2d_56/Conv2D/ReadVariableOpReadVariableOp(conv2d_56_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_56/Conv2DConv2D activation_53/Relu:activations:0'conv2d_56/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
?
 conv2d_56/BiasAdd/ReadVariableOpReadVariableOp)conv2d_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_56/BiasAddBiasAddconv2d_56/Conv2D:output:0(conv2d_56/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????t
conv2d_56/SigmoidSigmoidconv2d_56/BiasAdd:output:0*
T0*1
_output_shapes
:???????????n
IdentityIdentityconv2d_56/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp&^batch_normalization_37/AssignNewValue(^batch_normalization_37/AssignNewValue_17^batch_normalization_37/FusedBatchNormV3/ReadVariableOp9^batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_37/ReadVariableOp(^batch_normalization_37/ReadVariableOp_1&^batch_normalization_39/AssignNewValue(^batch_normalization_39/AssignNewValue_17^batch_normalization_39/FusedBatchNormV3/ReadVariableOp9^batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_39/ReadVariableOp(^batch_normalization_39/ReadVariableOp_1&^batch_normalization_41/AssignNewValue(^batch_normalization_41/AssignNewValue_17^batch_normalization_41/FusedBatchNormV3/ReadVariableOp9^batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_41/ReadVariableOp(^batch_normalization_41/ReadVariableOp_1&^batch_normalization_43/AssignNewValue(^batch_normalization_43/AssignNewValue_17^batch_normalization_43/FusedBatchNormV3/ReadVariableOp9^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_43/ReadVariableOp(^batch_normalization_43/ReadVariableOp_1&^batch_normalization_45/AssignNewValue(^batch_normalization_45/AssignNewValue_17^batch_normalization_45/FusedBatchNormV3/ReadVariableOp9^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_45/ReadVariableOp(^batch_normalization_45/ReadVariableOp_1&^batch_normalization_47/AssignNewValue(^batch_normalization_47/AssignNewValue_17^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_1&^batch_normalization_49/AssignNewValue(^batch_normalization_49/AssignNewValue_17^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_1&^batch_normalization_51/AssignNewValue(^batch_normalization_51/AssignNewValue_17^batch_normalization_51/FusedBatchNormV3/ReadVariableOp9^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_51/ReadVariableOp(^batch_normalization_51/ReadVariableOp_1&^batch_normalization_53/AssignNewValue(^batch_normalization_53/AssignNewValue_17^batch_normalization_53/FusedBatchNormV3/ReadVariableOp9^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_53/ReadVariableOp(^batch_normalization_53/ReadVariableOp_1!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp!^conv2d_55/BiasAdd/ReadVariableOp ^conv2d_55/Conv2D/ReadVariableOp!^conv2d_56/BiasAdd/ReadVariableOp ^conv2d_56/Conv2D/ReadVariableOp+^conv2d_transpose_10/BiasAdd/ReadVariableOp4^conv2d_transpose_10/conv2d_transpose/ReadVariableOp+^conv2d_transpose_11/BiasAdd/ReadVariableOp4^conv2d_transpose_11/conv2d_transpose/ReadVariableOp*^conv2d_transpose_8/BiasAdd/ReadVariableOp3^conv2d_transpose_8/conv2d_transpose/ReadVariableOp*^conv2d_transpose_9/BiasAdd/ReadVariableOp3^conv2d_transpose_9/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_37/AssignNewValue%batch_normalization_37/AssignNewValue2R
'batch_normalization_37/AssignNewValue_1'batch_normalization_37/AssignNewValue_12p
6batch_normalization_37/FusedBatchNormV3/ReadVariableOp6batch_normalization_37/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_18batch_normalization_37/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_37/ReadVariableOp%batch_normalization_37/ReadVariableOp2R
'batch_normalization_37/ReadVariableOp_1'batch_normalization_37/ReadVariableOp_12N
%batch_normalization_39/AssignNewValue%batch_normalization_39/AssignNewValue2R
'batch_normalization_39/AssignNewValue_1'batch_normalization_39/AssignNewValue_12p
6batch_normalization_39/FusedBatchNormV3/ReadVariableOp6batch_normalization_39/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_18batch_normalization_39/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_39/ReadVariableOp%batch_normalization_39/ReadVariableOp2R
'batch_normalization_39/ReadVariableOp_1'batch_normalization_39/ReadVariableOp_12N
%batch_normalization_41/AssignNewValue%batch_normalization_41/AssignNewValue2R
'batch_normalization_41/AssignNewValue_1'batch_normalization_41/AssignNewValue_12p
6batch_normalization_41/FusedBatchNormV3/ReadVariableOp6batch_normalization_41/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_18batch_normalization_41/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_41/ReadVariableOp%batch_normalization_41/ReadVariableOp2R
'batch_normalization_41/ReadVariableOp_1'batch_normalization_41/ReadVariableOp_12N
%batch_normalization_43/AssignNewValue%batch_normalization_43/AssignNewValue2R
'batch_normalization_43/AssignNewValue_1'batch_normalization_43/AssignNewValue_12p
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp6batch_normalization_43/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_18batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_43/ReadVariableOp%batch_normalization_43/ReadVariableOp2R
'batch_normalization_43/ReadVariableOp_1'batch_normalization_43/ReadVariableOp_12N
%batch_normalization_45/AssignNewValue%batch_normalization_45/AssignNewValue2R
'batch_normalization_45/AssignNewValue_1'batch_normalization_45/AssignNewValue_12p
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp6batch_normalization_45/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_18batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_45/ReadVariableOp%batch_normalization_45/ReadVariableOp2R
'batch_normalization_45/ReadVariableOp_1'batch_normalization_45/ReadVariableOp_12N
%batch_normalization_47/AssignNewValue%batch_normalization_47/AssignNewValue2R
'batch_normalization_47/AssignNewValue_1'batch_normalization_47/AssignNewValue_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12N
%batch_normalization_49/AssignNewValue%batch_normalization_49/AssignNewValue2R
'batch_normalization_49/AssignNewValue_1'batch_normalization_49/AssignNewValue_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12N
%batch_normalization_51/AssignNewValue%batch_normalization_51/AssignNewValue2R
'batch_normalization_51/AssignNewValue_1'batch_normalization_51/AssignNewValue_12p
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp6batch_normalization_51/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_18batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_51/ReadVariableOp%batch_normalization_51/ReadVariableOp2R
'batch_normalization_51/ReadVariableOp_1'batch_normalization_51/ReadVariableOp_12N
%batch_normalization_53/AssignNewValue%batch_normalization_53/AssignNewValue2R
'batch_normalization_53/AssignNewValue_1'batch_normalization_53/AssignNewValue_12p
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp6batch_normalization_53/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_18batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_53/ReadVariableOp%batch_normalization_53/ReadVariableOp2R
'batch_normalization_53/ReadVariableOp_1'batch_normalization_53/ReadVariableOp_12D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2D
 conv2d_55/BiasAdd/ReadVariableOp conv2d_55/BiasAdd/ReadVariableOp2B
conv2d_55/Conv2D/ReadVariableOpconv2d_55/Conv2D/ReadVariableOp2D
 conv2d_56/BiasAdd/ReadVariableOp conv2d_56/BiasAdd/ReadVariableOp2B
conv2d_56/Conv2D/ReadVariableOpconv2d_56/Conv2D/ReadVariableOp2X
*conv2d_transpose_10/BiasAdd/ReadVariableOp*conv2d_transpose_10/BiasAdd/ReadVariableOp2j
3conv2d_transpose_10/conv2d_transpose/ReadVariableOp3conv2d_transpose_10/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_11/BiasAdd/ReadVariableOp*conv2d_transpose_11/BiasAdd/ReadVariableOp2j
3conv2d_transpose_11/conv2d_transpose/ReadVariableOp3conv2d_transpose_11/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_8/BiasAdd/ReadVariableOp)conv2d_transpose_8/BiasAdd/ReadVariableOp2h
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_9/BiasAdd/ReadVariableOp)conv2d_transpose_9/BiasAdd/ReadVariableOp2h
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2conv2d_transpose_9/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_11_layer_call_fn_85232

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_81110?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
F
*__inference_dropout_22_layer_call_fn_85110

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_22_layer_call_and_return_conditional_losses_81505j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?

d
E__inference_dropout_22_layer_call_and_return_conditional_losses_85132

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??n
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@y
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@s
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@c
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
Z
.__inference_concatenate_11_layer_call_fn_85271
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_11_layer_call_and_return_conditional_losses_81551j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_80434

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_84330

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_41_layer_call_fn_84238

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_81238y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:??????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_39_layer_call_fn_84261

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_80479?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_49_layer_call_and_return_conditional_losses_84805

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

?
D__inference_conv2d_43_layer_call_and_return_conditional_losses_84376

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
c
E__inference_dropout_21_layer_call_and_return_conditional_losses_84947

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????@@?d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????@@?"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_85195

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_53_layer_call_and_return_conditional_losses_85151

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_47_layer_call_fn_84622

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_81358x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

d
E__inference_dropout_16_layer_call_and_return_conditional_losses_84229

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??n
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????y
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????s
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????c
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_9_layer_call_fn_84925
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_9_layer_call_and_return_conditional_losses_81445i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????@@?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????@@@:?????????@@@:Y U
/
_output_shapes
:?????????@@@
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????@@@
"
_user_specified_name
inputs/1
?

?
D__inference_conv2d_53_layer_call_and_return_conditional_losses_81517

inputs8
conv2d_readvariableop_resource:@ -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_37_layer_call_fn_84133

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_80403?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_84202

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_45_layer_call_and_return_conditional_losses_84704

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:??????????c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_80682

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_84310

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
s
I__inference_concatenate_10_layer_call_and_return_conditional_losses_81498

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????? :??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs:YU
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
c
E__inference_dropout_17_layer_call_and_return_conditional_losses_84345

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@@ c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@@ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
)__inference_conv2d_51_layer_call_fn_84968

inputs"
unknown:?@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_51_layer_call_and_return_conditional_losses_81464w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_45_layer_call_fn_84645

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80707?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_51_layer_call_fn_85177

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_81062?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_37_layer_call_fn_84146

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_80434?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_84566

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_47_layer_call_and_return_conditional_losses_84877

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????  ?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

?
D__inference_conv2d_43_layer_call_and_return_conditional_losses_81278

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_39_layer_call_and_return_conditional_losses_84120

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_23_layer_call_fn_85283

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_23_layer_call_and_return_conditional_losses_81558j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_81031

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_8_layer_call_fn_84713

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_80786?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80846

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80815

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_85040

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
I
-__inference_activation_53_layer_call_fn_85391

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_53_layer_call_and_return_conditional_losses_81590j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_84586

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

d
E__inference_dropout_18_layer_call_and_return_conditional_losses_84485

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????  @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????  @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  @w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  @q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  @a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
)__inference_conv2d_49_layer_call_fn_84795

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_81411x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????  ?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

d
E__inference_dropout_21_layer_call_and_return_conditional_losses_81879

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????@@?C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????@@?*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????@@?x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????@@?r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????@@?b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????@@?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
'__inference_model_2_layer_call_fn_83453

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?%

unknown_37:@?

unknown_38:@%

unknown_39:?@

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@$

unknown_45: @

unknown_46: $

unknown_47:@ 

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: $

unknown_53: 

unknown_54:$

unknown_55: 

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:$

unknown_61:

unknown_62:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*P
_read_only_resource_inputs2
0.	
 !"#$'()*+,/01234789:;<?@*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_82422y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
u
I__inference_concatenate_11_layer_call_and_return_conditional_losses_85278
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:??????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?	
?
6__inference_batch_normalization_41_layer_call_fn_84402

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_80586?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_8_layer_call_fn_84197

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_80454?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_39_layer_call_and_return_conditional_losses_84320

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:??????????? d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_51_layer_call_and_return_conditional_losses_84978

inputs9
conv2d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
? 
?
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_80786

inputsD
(conv2d_transpose_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_47_layer_call_and_return_conditional_losses_81358

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_23_layer_call_and_return_conditional_losses_81558

inputs

identity_1X
IdentityIdentityinputs*
T0*1
_output_shapes
:??????????? e

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:??????????? "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
d
H__inference_activation_43_layer_call_and_return_conditional_losses_81338

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????  ?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
c
E__inference_dropout_17_layer_call_and_return_conditional_losses_81266

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????@@ c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????@@ "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
c
*__inference_dropout_19_layer_call_fn_84596

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_81971x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_49_layer_call_fn_85004

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80954?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

?
D__inference_conv2d_41_layer_call_and_return_conditional_losses_84248

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80923

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
I
-__inference_activation_39_layer_call_fn_84315

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_39_layer_call_and_return_conditional_losses_81258j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_9_layer_call_fn_84325

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_80530?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_55_layer_call_and_return_conditional_losses_81570

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
I
-__inference_activation_51_layer_call_fn_85218

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_51_layer_call_and_return_conditional_losses_81537j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:??????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_80662

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_41_layer_call_and_return_conditional_losses_81238

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:??????????? w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_17_layer_call_fn_84335

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_81266h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@ "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@ :W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_84867

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
*__inference_dropout_18_layer_call_fn_84468

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_82010w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????  @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_84849

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
d
H__inference_activation_47_layer_call_and_return_conditional_losses_81431

inputs
identityO
ReluReluinputs*
T0*0
_output_shapes
:?????????  ?c
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
Ǥ
?r
!__inference__traced_restore_86439
file_prefix;
!assignvariableop_conv2d_39_kernel:/
!assignvariableop_1_conv2d_39_bias:=
/assignvariableop_2_batch_normalization_37_gamma:<
.assignvariableop_3_batch_normalization_37_beta:C
5assignvariableop_4_batch_normalization_37_moving_mean:G
9assignvariableop_5_batch_normalization_37_moving_variance:=
#assignvariableop_6_conv2d_41_kernel: /
!assignvariableop_7_conv2d_41_bias: =
/assignvariableop_8_batch_normalization_39_gamma: <
.assignvariableop_9_batch_normalization_39_beta: D
6assignvariableop_10_batch_normalization_39_moving_mean: H
:assignvariableop_11_batch_normalization_39_moving_variance: >
$assignvariableop_12_conv2d_43_kernel: @0
"assignvariableop_13_conv2d_43_bias:@>
0assignvariableop_14_batch_normalization_41_gamma:@=
/assignvariableop_15_batch_normalization_41_beta:@D
6assignvariableop_16_batch_normalization_41_moving_mean:@H
:assignvariableop_17_batch_normalization_41_moving_variance:@?
$assignvariableop_18_conv2d_45_kernel:@?1
"assignvariableop_19_conv2d_45_bias:	??
0assignvariableop_20_batch_normalization_43_gamma:	?>
/assignvariableop_21_batch_normalization_43_beta:	?E
6assignvariableop_22_batch_normalization_43_moving_mean:	?I
:assignvariableop_23_batch_normalization_43_moving_variance:	?@
$assignvariableop_24_conv2d_47_kernel:??1
"assignvariableop_25_conv2d_47_bias:	??
0assignvariableop_26_batch_normalization_45_gamma:	?>
/assignvariableop_27_batch_normalization_45_beta:	?E
6assignvariableop_28_batch_normalization_45_moving_mean:	?I
:assignvariableop_29_batch_normalization_45_moving_variance:	?I
-assignvariableop_30_conv2d_transpose_8_kernel:??:
+assignvariableop_31_conv2d_transpose_8_bias:	?@
$assignvariableop_32_conv2d_49_kernel:??1
"assignvariableop_33_conv2d_49_bias:	??
0assignvariableop_34_batch_normalization_47_gamma:	?>
/assignvariableop_35_batch_normalization_47_beta:	?E
6assignvariableop_36_batch_normalization_47_moving_mean:	?I
:assignvariableop_37_batch_normalization_47_moving_variance:	?H
-assignvariableop_38_conv2d_transpose_9_kernel:@?9
+assignvariableop_39_conv2d_transpose_9_bias:@?
$assignvariableop_40_conv2d_51_kernel:?@0
"assignvariableop_41_conv2d_51_bias:@>
0assignvariableop_42_batch_normalization_49_gamma:@=
/assignvariableop_43_batch_normalization_49_beta:@D
6assignvariableop_44_batch_normalization_49_moving_mean:@H
:assignvariableop_45_batch_normalization_49_moving_variance:@H
.assignvariableop_46_conv2d_transpose_10_kernel: @:
,assignvariableop_47_conv2d_transpose_10_bias: >
$assignvariableop_48_conv2d_53_kernel:@ 0
"assignvariableop_49_conv2d_53_bias: >
0assignvariableop_50_batch_normalization_51_gamma: =
/assignvariableop_51_batch_normalization_51_beta: D
6assignvariableop_52_batch_normalization_51_moving_mean: H
:assignvariableop_53_batch_normalization_51_moving_variance: H
.assignvariableop_54_conv2d_transpose_11_kernel: :
,assignvariableop_55_conv2d_transpose_11_bias:>
$assignvariableop_56_conv2d_55_kernel: 0
"assignvariableop_57_conv2d_55_bias:>
0assignvariableop_58_batch_normalization_53_gamma:=
/assignvariableop_59_batch_normalization_53_beta:D
6assignvariableop_60_batch_normalization_53_moving_mean:H
:assignvariableop_61_batch_normalization_53_moving_variance:>
$assignvariableop_62_conv2d_56_kernel:0
"assignvariableop_63_conv2d_56_bias:'
assignvariableop_64_adam_iter:	 )
assignvariableop_65_adam_beta_1: )
assignvariableop_66_adam_beta_2: (
assignvariableop_67_adam_decay: 0
&assignvariableop_68_adam_learning_rate: %
assignvariableop_69_total_1: %
assignvariableop_70_count_1: #
assignvariableop_71_total: #
assignvariableop_72_count: E
+assignvariableop_73_adam_conv2d_39_kernel_m:7
)assignvariableop_74_adam_conv2d_39_bias_m:E
7assignvariableop_75_adam_batch_normalization_37_gamma_m:D
6assignvariableop_76_adam_batch_normalization_37_beta_m:E
+assignvariableop_77_adam_conv2d_41_kernel_m: 7
)assignvariableop_78_adam_conv2d_41_bias_m: E
7assignvariableop_79_adam_batch_normalization_39_gamma_m: D
6assignvariableop_80_adam_batch_normalization_39_beta_m: E
+assignvariableop_81_adam_conv2d_43_kernel_m: @7
)assignvariableop_82_adam_conv2d_43_bias_m:@E
7assignvariableop_83_adam_batch_normalization_41_gamma_m:@D
6assignvariableop_84_adam_batch_normalization_41_beta_m:@F
+assignvariableop_85_adam_conv2d_45_kernel_m:@?8
)assignvariableop_86_adam_conv2d_45_bias_m:	?F
7assignvariableop_87_adam_batch_normalization_43_gamma_m:	?E
6assignvariableop_88_adam_batch_normalization_43_beta_m:	?G
+assignvariableop_89_adam_conv2d_47_kernel_m:??8
)assignvariableop_90_adam_conv2d_47_bias_m:	?F
7assignvariableop_91_adam_batch_normalization_45_gamma_m:	?E
6assignvariableop_92_adam_batch_normalization_45_beta_m:	?P
4assignvariableop_93_adam_conv2d_transpose_8_kernel_m:??A
2assignvariableop_94_adam_conv2d_transpose_8_bias_m:	?G
+assignvariableop_95_adam_conv2d_49_kernel_m:??8
)assignvariableop_96_adam_conv2d_49_bias_m:	?F
7assignvariableop_97_adam_batch_normalization_47_gamma_m:	?E
6assignvariableop_98_adam_batch_normalization_47_beta_m:	?O
4assignvariableop_99_adam_conv2d_transpose_9_kernel_m:@?A
3assignvariableop_100_adam_conv2d_transpose_9_bias_m:@G
,assignvariableop_101_adam_conv2d_51_kernel_m:?@8
*assignvariableop_102_adam_conv2d_51_bias_m:@F
8assignvariableop_103_adam_batch_normalization_49_gamma_m:@E
7assignvariableop_104_adam_batch_normalization_49_beta_m:@P
6assignvariableop_105_adam_conv2d_transpose_10_kernel_m: @B
4assignvariableop_106_adam_conv2d_transpose_10_bias_m: F
,assignvariableop_107_adam_conv2d_53_kernel_m:@ 8
*assignvariableop_108_adam_conv2d_53_bias_m: F
8assignvariableop_109_adam_batch_normalization_51_gamma_m: E
7assignvariableop_110_adam_batch_normalization_51_beta_m: P
6assignvariableop_111_adam_conv2d_transpose_11_kernel_m: B
4assignvariableop_112_adam_conv2d_transpose_11_bias_m:F
,assignvariableop_113_adam_conv2d_55_kernel_m: 8
*assignvariableop_114_adam_conv2d_55_bias_m:F
8assignvariableop_115_adam_batch_normalization_53_gamma_m:E
7assignvariableop_116_adam_batch_normalization_53_beta_m:F
,assignvariableop_117_adam_conv2d_56_kernel_m:8
*assignvariableop_118_adam_conv2d_56_bias_m:F
,assignvariableop_119_adam_conv2d_39_kernel_v:8
*assignvariableop_120_adam_conv2d_39_bias_v:F
8assignvariableop_121_adam_batch_normalization_37_gamma_v:E
7assignvariableop_122_adam_batch_normalization_37_beta_v:F
,assignvariableop_123_adam_conv2d_41_kernel_v: 8
*assignvariableop_124_adam_conv2d_41_bias_v: F
8assignvariableop_125_adam_batch_normalization_39_gamma_v: E
7assignvariableop_126_adam_batch_normalization_39_beta_v: F
,assignvariableop_127_adam_conv2d_43_kernel_v: @8
*assignvariableop_128_adam_conv2d_43_bias_v:@F
8assignvariableop_129_adam_batch_normalization_41_gamma_v:@E
7assignvariableop_130_adam_batch_normalization_41_beta_v:@G
,assignvariableop_131_adam_conv2d_45_kernel_v:@?9
*assignvariableop_132_adam_conv2d_45_bias_v:	?G
8assignvariableop_133_adam_batch_normalization_43_gamma_v:	?F
7assignvariableop_134_adam_batch_normalization_43_beta_v:	?H
,assignvariableop_135_adam_conv2d_47_kernel_v:??9
*assignvariableop_136_adam_conv2d_47_bias_v:	?G
8assignvariableop_137_adam_batch_normalization_45_gamma_v:	?F
7assignvariableop_138_adam_batch_normalization_45_beta_v:	?Q
5assignvariableop_139_adam_conv2d_transpose_8_kernel_v:??B
3assignvariableop_140_adam_conv2d_transpose_8_bias_v:	?H
,assignvariableop_141_adam_conv2d_49_kernel_v:??9
*assignvariableop_142_adam_conv2d_49_bias_v:	?G
8assignvariableop_143_adam_batch_normalization_47_gamma_v:	?F
7assignvariableop_144_adam_batch_normalization_47_beta_v:	?P
5assignvariableop_145_adam_conv2d_transpose_9_kernel_v:@?A
3assignvariableop_146_adam_conv2d_transpose_9_bias_v:@G
,assignvariableop_147_adam_conv2d_51_kernel_v:?@8
*assignvariableop_148_adam_conv2d_51_bias_v:@F
8assignvariableop_149_adam_batch_normalization_49_gamma_v:@E
7assignvariableop_150_adam_batch_normalization_49_beta_v:@P
6assignvariableop_151_adam_conv2d_transpose_10_kernel_v: @B
4assignvariableop_152_adam_conv2d_transpose_10_bias_v: F
,assignvariableop_153_adam_conv2d_53_kernel_v:@ 8
*assignvariableop_154_adam_conv2d_53_bias_v: F
8assignvariableop_155_adam_batch_normalization_51_gamma_v: E
7assignvariableop_156_adam_batch_normalization_51_beta_v: P
6assignvariableop_157_adam_conv2d_transpose_11_kernel_v: B
4assignvariableop_158_adam_conv2d_transpose_11_bias_v:F
,assignvariableop_159_adam_conv2d_55_kernel_v: 8
*assignvariableop_160_adam_conv2d_55_bias_v:F
8assignvariableop_161_adam_batch_normalization_53_gamma_v:E
7assignvariableop_162_adam_batch_normalization_53_beta_v:F
,assignvariableop_163_adam_conv2d_56_kernel_v:8
*assignvariableop_164_adam_conv2d_56_bias_v:
identity_166??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_103?AssignVariableOp_104?AssignVariableOp_105?AssignVariableOp_106?AssignVariableOp_107?AssignVariableOp_108?AssignVariableOp_109?AssignVariableOp_11?AssignVariableOp_110?AssignVariableOp_111?AssignVariableOp_112?AssignVariableOp_113?AssignVariableOp_114?AssignVariableOp_115?AssignVariableOp_116?AssignVariableOp_117?AssignVariableOp_118?AssignVariableOp_119?AssignVariableOp_12?AssignVariableOp_120?AssignVariableOp_121?AssignVariableOp_122?AssignVariableOp_123?AssignVariableOp_124?AssignVariableOp_125?AssignVariableOp_126?AssignVariableOp_127?AssignVariableOp_128?AssignVariableOp_129?AssignVariableOp_13?AssignVariableOp_130?AssignVariableOp_131?AssignVariableOp_132?AssignVariableOp_133?AssignVariableOp_134?AssignVariableOp_135?AssignVariableOp_136?AssignVariableOp_137?AssignVariableOp_138?AssignVariableOp_139?AssignVariableOp_14?AssignVariableOp_140?AssignVariableOp_141?AssignVariableOp_142?AssignVariableOp_143?AssignVariableOp_144?AssignVariableOp_145?AssignVariableOp_146?AssignVariableOp_147?AssignVariableOp_148?AssignVariableOp_149?AssignVariableOp_15?AssignVariableOp_150?AssignVariableOp_151?AssignVariableOp_152?AssignVariableOp_153?AssignVariableOp_154?AssignVariableOp_155?AssignVariableOp_156?AssignVariableOp_157?AssignVariableOp_158?AssignVariableOp_159?AssignVariableOp_16?AssignVariableOp_160?AssignVariableOp_161?AssignVariableOp_162?AssignVariableOp_163?AssignVariableOp_164?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?]
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?\
value?\B?\?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-18/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-18/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-18/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-19/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-19/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-20/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-20/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-21/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-21/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-21/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-21/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-22/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-22/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-21/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-18/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-19/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-19/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-20/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-20/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-21/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-21/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-22/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-22/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:?*
dtype0*?
value?B??B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*?
dtypes?
?2?	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_39_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_39_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_37_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_37_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_37_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_37_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_41_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_41_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_39_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_39_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_39_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_39_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_43_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_43_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_41_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_41_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_41_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_41_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp$assignvariableop_18_conv2d_45_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp"assignvariableop_19_conv2d_45_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp0assignvariableop_20_batch_normalization_43_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_43_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp6assignvariableop_22_batch_normalization_43_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp:assignvariableop_23_batch_normalization_43_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp$assignvariableop_24_conv2d_47_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp"assignvariableop_25_conv2d_47_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_26AssignVariableOp0assignvariableop_26_batch_normalization_45_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOp/assignvariableop_27_batch_normalization_45_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOp6assignvariableop_28_batch_normalization_45_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOp:assignvariableop_29_batch_normalization_45_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp-assignvariableop_30_conv2d_transpose_8_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOp+assignvariableop_31_conv2d_transpose_8_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOp$assignvariableop_32_conv2d_49_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOp"assignvariableop_33_conv2d_49_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOp0assignvariableop_34_batch_normalization_47_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp/assignvariableop_35_batch_normalization_47_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp6assignvariableop_36_batch_normalization_47_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp:assignvariableop_37_batch_normalization_47_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp-assignvariableop_38_conv2d_transpose_9_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp+assignvariableop_39_conv2d_transpose_9_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp$assignvariableop_40_conv2d_51_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp"assignvariableop_41_conv2d_51_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp0assignvariableop_42_batch_normalization_49_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp/assignvariableop_43_batch_normalization_49_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp6assignvariableop_44_batch_normalization_49_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp:assignvariableop_45_batch_normalization_49_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp.assignvariableop_46_conv2d_transpose_10_kernelIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp,assignvariableop_47_conv2d_transpose_10_biasIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp$assignvariableop_48_conv2d_53_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp"assignvariableop_49_conv2d_53_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp0assignvariableop_50_batch_normalization_51_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp/assignvariableop_51_batch_normalization_51_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp6assignvariableop_52_batch_normalization_51_moving_meanIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp:assignvariableop_53_batch_normalization_51_moving_varianceIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp.assignvariableop_54_conv2d_transpose_11_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp,assignvariableop_55_conv2d_transpose_11_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp$assignvariableop_56_conv2d_55_kernelIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp"assignvariableop_57_conv2d_55_biasIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp0assignvariableop_58_batch_normalization_53_gammaIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp/assignvariableop_59_batch_normalization_53_betaIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp6assignvariableop_60_batch_normalization_53_moving_meanIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp:assignvariableop_61_batch_normalization_53_moving_varianceIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp$assignvariableop_62_conv2d_56_kernelIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp"assignvariableop_63_conv2d_56_biasIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_64AssignVariableOpassignvariableop_64_adam_iterIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOpassignvariableop_65_adam_beta_1Identity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOpassignvariableop_66_adam_beta_2Identity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOpassignvariableop_67_adam_decayIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp&assignvariableop_68_adam_learning_rateIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOpassignvariableop_69_total_1Identity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOpassignvariableop_70_count_1Identity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOpassignvariableop_71_totalIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOpassignvariableop_72_countIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp+assignvariableop_73_adam_conv2d_39_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp)assignvariableop_74_adam_conv2d_39_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_75AssignVariableOp7assignvariableop_75_adam_batch_normalization_37_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_76AssignVariableOp6assignvariableop_76_adam_batch_normalization_37_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_77AssignVariableOp+assignvariableop_77_adam_conv2d_41_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_78AssignVariableOp)assignvariableop_78_adam_conv2d_41_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_79AssignVariableOp7assignvariableop_79_adam_batch_normalization_39_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_80AssignVariableOp6assignvariableop_80_adam_batch_normalization_39_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_81AssignVariableOp+assignvariableop_81_adam_conv2d_43_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_82AssignVariableOp)assignvariableop_82_adam_conv2d_43_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_83AssignVariableOp7assignvariableop_83_adam_batch_normalization_41_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_84AssignVariableOp6assignvariableop_84_adam_batch_normalization_41_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_85AssignVariableOp+assignvariableop_85_adam_conv2d_45_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_86AssignVariableOp)assignvariableop_86_adam_conv2d_45_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_87AssignVariableOp7assignvariableop_87_adam_batch_normalization_43_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_88AssignVariableOp6assignvariableop_88_adam_batch_normalization_43_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_89AssignVariableOp+assignvariableop_89_adam_conv2d_47_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_90AssignVariableOp)assignvariableop_90_adam_conv2d_47_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_91AssignVariableOp7assignvariableop_91_adam_batch_normalization_45_gamma_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_92AssignVariableOp6assignvariableop_92_adam_batch_normalization_45_beta_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_93AssignVariableOp4assignvariableop_93_adam_conv2d_transpose_8_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_94AssignVariableOp2assignvariableop_94_adam_conv2d_transpose_8_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_95AssignVariableOp+assignvariableop_95_adam_conv2d_49_kernel_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_96AssignVariableOp)assignvariableop_96_adam_conv2d_49_bias_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_97AssignVariableOp7assignvariableop_97_adam_batch_normalization_47_gamma_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_98AssignVariableOp6assignvariableop_98_adam_batch_normalization_47_beta_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_99AssignVariableOp4assignvariableop_99_adam_conv2d_transpose_9_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_100AssignVariableOp3assignvariableop_100_adam_conv2d_transpose_9_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_101AssignVariableOp,assignvariableop_101_adam_conv2d_51_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_102AssignVariableOp*assignvariableop_102_adam_conv2d_51_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_103AssignVariableOp8assignvariableop_103_adam_batch_normalization_49_gamma_mIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_104AssignVariableOp7assignvariableop_104_adam_batch_normalization_49_beta_mIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_105AssignVariableOp6assignvariableop_105_adam_conv2d_transpose_10_kernel_mIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_106AssignVariableOp4assignvariableop_106_adam_conv2d_transpose_10_bias_mIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_107AssignVariableOp,assignvariableop_107_adam_conv2d_53_kernel_mIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_108AssignVariableOp*assignvariableop_108_adam_conv2d_53_bias_mIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_109AssignVariableOp8assignvariableop_109_adam_batch_normalization_51_gamma_mIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_110AssignVariableOp7assignvariableop_110_adam_batch_normalization_51_beta_mIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_111AssignVariableOp6assignvariableop_111_adam_conv2d_transpose_11_kernel_mIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_112AssignVariableOp4assignvariableop_112_adam_conv2d_transpose_11_bias_mIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_113AssignVariableOp,assignvariableop_113_adam_conv2d_55_kernel_mIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_114AssignVariableOp*assignvariableop_114_adam_conv2d_55_bias_mIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_115AssignVariableOp8assignvariableop_115_adam_batch_normalization_53_gamma_mIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_116AssignVariableOp7assignvariableop_116_adam_batch_normalization_53_beta_mIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_117AssignVariableOp,assignvariableop_117_adam_conv2d_56_kernel_mIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_118AssignVariableOp*assignvariableop_118_adam_conv2d_56_bias_mIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_119AssignVariableOp,assignvariableop_119_adam_conv2d_39_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_120AssignVariableOp*assignvariableop_120_adam_conv2d_39_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_121AssignVariableOp8assignvariableop_121_adam_batch_normalization_37_gamma_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_122AssignVariableOp7assignvariableop_122_adam_batch_normalization_37_beta_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_123AssignVariableOp,assignvariableop_123_adam_conv2d_41_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_124AssignVariableOp*assignvariableop_124_adam_conv2d_41_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_125AssignVariableOp8assignvariableop_125_adam_batch_normalization_39_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_126AssignVariableOp7assignvariableop_126_adam_batch_normalization_39_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_127AssignVariableOp,assignvariableop_127_adam_conv2d_43_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_128AssignVariableOp*assignvariableop_128_adam_conv2d_43_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_129AssignVariableOp8assignvariableop_129_adam_batch_normalization_41_gamma_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_130AssignVariableOp7assignvariableop_130_adam_batch_normalization_41_beta_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_131AssignVariableOp,assignvariableop_131_adam_conv2d_45_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_132AssignVariableOp*assignvariableop_132_adam_conv2d_45_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_133AssignVariableOp8assignvariableop_133_adam_batch_normalization_43_gamma_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_134AssignVariableOp7assignvariableop_134_adam_batch_normalization_43_beta_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_135AssignVariableOp,assignvariableop_135_adam_conv2d_47_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_136AssignVariableOp*assignvariableop_136_adam_conv2d_47_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_137AssignVariableOp8assignvariableop_137_adam_batch_normalization_45_gamma_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_138AssignVariableOp7assignvariableop_138_adam_batch_normalization_45_beta_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_139AssignVariableOp5assignvariableop_139_adam_conv2d_transpose_8_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_140AssignVariableOp3assignvariableop_140_adam_conv2d_transpose_8_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_141IdentityRestoreV2:tensors:141"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_141AssignVariableOp,assignvariableop_141_adam_conv2d_49_kernel_vIdentity_141:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_142IdentityRestoreV2:tensors:142"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_142AssignVariableOp*assignvariableop_142_adam_conv2d_49_bias_vIdentity_142:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_143IdentityRestoreV2:tensors:143"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_143AssignVariableOp8assignvariableop_143_adam_batch_normalization_47_gamma_vIdentity_143:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_144IdentityRestoreV2:tensors:144"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_144AssignVariableOp7assignvariableop_144_adam_batch_normalization_47_beta_vIdentity_144:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_145IdentityRestoreV2:tensors:145"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_145AssignVariableOp5assignvariableop_145_adam_conv2d_transpose_9_kernel_vIdentity_145:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_146IdentityRestoreV2:tensors:146"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_146AssignVariableOp3assignvariableop_146_adam_conv2d_transpose_9_bias_vIdentity_146:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_147IdentityRestoreV2:tensors:147"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_147AssignVariableOp,assignvariableop_147_adam_conv2d_51_kernel_vIdentity_147:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_148IdentityRestoreV2:tensors:148"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_148AssignVariableOp*assignvariableop_148_adam_conv2d_51_bias_vIdentity_148:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_149IdentityRestoreV2:tensors:149"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_149AssignVariableOp8assignvariableop_149_adam_batch_normalization_49_gamma_vIdentity_149:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_150IdentityRestoreV2:tensors:150"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_150AssignVariableOp7assignvariableop_150_adam_batch_normalization_49_beta_vIdentity_150:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_151IdentityRestoreV2:tensors:151"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_151AssignVariableOp6assignvariableop_151_adam_conv2d_transpose_10_kernel_vIdentity_151:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_152IdentityRestoreV2:tensors:152"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_152AssignVariableOp4assignvariableop_152_adam_conv2d_transpose_10_bias_vIdentity_152:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_153IdentityRestoreV2:tensors:153"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_153AssignVariableOp,assignvariableop_153_adam_conv2d_53_kernel_vIdentity_153:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_154IdentityRestoreV2:tensors:154"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_154AssignVariableOp*assignvariableop_154_adam_conv2d_53_bias_vIdentity_154:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_155IdentityRestoreV2:tensors:155"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_155AssignVariableOp8assignvariableop_155_adam_batch_normalization_51_gamma_vIdentity_155:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_156IdentityRestoreV2:tensors:156"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_156AssignVariableOp7assignvariableop_156_adam_batch_normalization_51_beta_vIdentity_156:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_157IdentityRestoreV2:tensors:157"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_157AssignVariableOp6assignvariableop_157_adam_conv2d_transpose_11_kernel_vIdentity_157:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_158IdentityRestoreV2:tensors:158"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_158AssignVariableOp4assignvariableop_158_adam_conv2d_transpose_11_bias_vIdentity_158:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_159IdentityRestoreV2:tensors:159"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_159AssignVariableOp,assignvariableop_159_adam_conv2d_55_kernel_vIdentity_159:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_160IdentityRestoreV2:tensors:160"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_160AssignVariableOp*assignvariableop_160_adam_conv2d_55_bias_vIdentity_160:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_161IdentityRestoreV2:tensors:161"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_161AssignVariableOp8assignvariableop_161_adam_batch_normalization_53_gamma_vIdentity_161:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_162IdentityRestoreV2:tensors:162"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_162AssignVariableOp7assignvariableop_162_adam_batch_normalization_53_beta_vIdentity_162:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_163IdentityRestoreV2:tensors:163"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_163AssignVariableOp,assignvariableop_163_adam_conv2d_56_kernel_vIdentity_163:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_164IdentityRestoreV2:tensors:164"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_164AssignVariableOp*assignvariableop_164_adam_conv2d_56_bias_vIdentity_164:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_165Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_166IdentityIdentity_165:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_141^AssignVariableOp_142^AssignVariableOp_143^AssignVariableOp_144^AssignVariableOp_145^AssignVariableOp_146^AssignVariableOp_147^AssignVariableOp_148^AssignVariableOp_149^AssignVariableOp_15^AssignVariableOp_150^AssignVariableOp_151^AssignVariableOp_152^AssignVariableOp_153^AssignVariableOp_154^AssignVariableOp_155^AssignVariableOp_156^AssignVariableOp_157^AssignVariableOp_158^AssignVariableOp_159^AssignVariableOp_16^AssignVariableOp_160^AssignVariableOp_161^AssignVariableOp_162^AssignVariableOp_163^AssignVariableOp_164^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_166Identity_166:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402,
AssignVariableOp_141AssignVariableOp_1412,
AssignVariableOp_142AssignVariableOp_1422,
AssignVariableOp_143AssignVariableOp_1432,
AssignVariableOp_144AssignVariableOp_1442,
AssignVariableOp_145AssignVariableOp_1452,
AssignVariableOp_146AssignVariableOp_1462,
AssignVariableOp_147AssignVariableOp_1472,
AssignVariableOp_148AssignVariableOp_1482,
AssignVariableOp_149AssignVariableOp_1492*
AssignVariableOp_15AssignVariableOp_152,
AssignVariableOp_150AssignVariableOp_1502,
AssignVariableOp_151AssignVariableOp_1512,
AssignVariableOp_152AssignVariableOp_1522,
AssignVariableOp_153AssignVariableOp_1532,
AssignVariableOp_154AssignVariableOp_1542,
AssignVariableOp_155AssignVariableOp_1552,
AssignVariableOp_156AssignVariableOp_1562,
AssignVariableOp_157AssignVariableOp_1572,
AssignVariableOp_158AssignVariableOp_1582,
AssignVariableOp_159AssignVariableOp_1592*
AssignVariableOp_16AssignVariableOp_162,
AssignVariableOp_160AssignVariableOp_1602,
AssignVariableOp_161AssignVariableOp_1612,
AssignVariableOp_162AssignVariableOp_1622,
AssignVariableOp_163AssignVariableOp_1632,
AssignVariableOp_164AssignVariableOp_1642*
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
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
D__inference_conv2d_56_layer_call_and_return_conditional_losses_85416

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????`
SigmoidSigmoidBiasAdd:output:0*
T0*1
_output_shapes
:???????????d
IdentityIdentitySigmoid:y:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_84676

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
? 
?
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_85092

inputsB
(conv2d_transpose_readvariableop_resource: @-
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B : y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
r
H__inference_concatenate_8_layer_call_and_return_conditional_losses_81392

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :~
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????  ?`
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:?????????  ?:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs:XT
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_80530

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_85213

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

?
D__inference_conv2d_51_layer_call_and_return_conditional_losses_81464

inputs9
conv2d_readvariableop_resource:?@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_81062

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_51_layer_call_fn_85164

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_81031?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_80555

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?

d
E__inference_dropout_18_layer_call_and_return_conditional_losses_82010

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????  @C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????  @*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????  @w
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????  @q
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????  @a
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_43_layer_call_fn_84530

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_80662?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_47_layer_call_and_return_conditional_losses_84632

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????h
IdentityIdentityBiasAdd:output:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_39_layer_call_fn_84110

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_81198y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_22_layer_call_and_return_conditional_losses_81505

inputs

identity_1X
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????@e

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
?
'__inference_model_2_layer_call_fn_81741
img!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?%

unknown_37:@?

unknown_38:@%

unknown_39:?@

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@$

unknown_45: @

unknown_46: $

unknown_47:@ 

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: $

unknown_53: 

unknown_54:$

unknown_55: 

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:$

unknown_61:

unknown_62:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallimgunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_81610y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
1
_output_shapes
:???????????

_user_specified_nameimg
?
?
)__inference_conv2d_56_layer_call_fn_85405

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_56_layer_call_and_return_conditional_losses_81603y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
H__inference_activation_53_layer_call_and_return_conditional_losses_81590

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:???????????d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_80403

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
I
-__inference_activation_43_layer_call_fn_84571

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_43_layer_call_and_return_conditional_losses_81338i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

d
E__inference_dropout_22_layer_call_and_return_conditional_losses_81833

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??n
dropout/MulMulinputsdropout/Const:output:0*
T0*1
_output_shapes
:???????????@C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*1
_output_shapes
:???????????@*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*1
_output_shapes
:???????????@y
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*1
_output_shapes
:???????????@s
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*1
_output_shapes
:???????????@c
IdentityIdentitydropout/Mul_1:z:0*
T0*1
_output_shapes
:???????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????@:Y U
1
_output_shapes
:???????????@
 
_user_specified_nameinputs
?
c
E__inference_dropout_19_layer_call_and_return_conditional_losses_84601

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:??????????d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:??????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_85386

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
F
*__inference_dropout_16_layer_call_fn_84207

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_81226j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
u
I__inference_concatenate_10_layer_call_and_return_conditional_losses_85105
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@a
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::??????????? :??????????? :[ W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:??????????? 
"
_user_specified_name
inputs/1
? 
?
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_84746

inputsD
(conv2d_transpose_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: J
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????z
IdentityIdentityBiasAdd:output:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_55_layer_call_and_return_conditional_losses_85324

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:??????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
F
*__inference_dropout_21_layer_call_fn_84937

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_81452i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????@@?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
c
E__inference_dropout_16_layer_call_and_return_conditional_losses_81226

inputs

identity_1X
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????e

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
? 
?
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_85265

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?:
B__inference_model_2_layer_call_and_return_conditional_losses_83749

inputsB
(conv2d_39_conv2d_readvariableop_resource:7
)conv2d_39_biasadd_readvariableop_resource:<
.batch_normalization_37_readvariableop_resource:>
0batch_normalization_37_readvariableop_1_resource:M
?batch_normalization_37_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_41_conv2d_readvariableop_resource: 7
)conv2d_41_biasadd_readvariableop_resource: <
.batch_normalization_39_readvariableop_resource: >
0batch_normalization_39_readvariableop_1_resource: M
?batch_normalization_39_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource: B
(conv2d_43_conv2d_readvariableop_resource: @7
)conv2d_43_biasadd_readvariableop_resource:@<
.batch_normalization_41_readvariableop_resource:@>
0batch_normalization_41_readvariableop_1_resource:@M
?batch_normalization_41_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_45_conv2d_readvariableop_resource:@?8
)conv2d_45_biasadd_readvariableop_resource:	?=
.batch_normalization_43_readvariableop_resource:	??
0batch_normalization_43_readvariableop_1_resource:	?N
?batch_normalization_43_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_47_conv2d_readvariableop_resource:??8
)conv2d_47_biasadd_readvariableop_resource:	?=
.batch_normalization_45_readvariableop_resource:	??
0batch_normalization_45_readvariableop_1_resource:	?N
?batch_normalization_45_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource:	?W
;conv2d_transpose_8_conv2d_transpose_readvariableop_resource:??A
2conv2d_transpose_8_biasadd_readvariableop_resource:	?D
(conv2d_49_conv2d_readvariableop_resource:??8
)conv2d_49_biasadd_readvariableop_resource:	?=
.batch_normalization_47_readvariableop_resource:	??
0batch_normalization_47_readvariableop_1_resource:	?N
?batch_normalization_47_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:	?V
;conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@?@
2conv2d_transpose_9_biasadd_readvariableop_resource:@C
(conv2d_51_conv2d_readvariableop_resource:?@7
)conv2d_51_biasadd_readvariableop_resource:@<
.batch_normalization_49_readvariableop_resource:@>
0batch_normalization_49_readvariableop_1_resource:@M
?batch_normalization_49_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource:@V
<conv2d_transpose_10_conv2d_transpose_readvariableop_resource: @A
3conv2d_transpose_10_biasadd_readvariableop_resource: B
(conv2d_53_conv2d_readvariableop_resource:@ 7
)conv2d_53_biasadd_readvariableop_resource: <
.batch_normalization_51_readvariableop_resource: >
0batch_normalization_51_readvariableop_1_resource: M
?batch_normalization_51_fusedbatchnormv3_readvariableop_resource: O
Abatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource: V
<conv2d_transpose_11_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_11_biasadd_readvariableop_resource:B
(conv2d_55_conv2d_readvariableop_resource: 7
)conv2d_55_biasadd_readvariableop_resource:<
.batch_normalization_53_readvariableop_resource:>
0batch_normalization_53_readvariableop_1_resource:M
?batch_normalization_53_fusedbatchnormv3_readvariableop_resource:O
Abatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource:B
(conv2d_56_conv2d_readvariableop_resource:7
)conv2d_56_biasadd_readvariableop_resource:
identity??6batch_normalization_37/FusedBatchNormV3/ReadVariableOp?8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_37/ReadVariableOp?'batch_normalization_37/ReadVariableOp_1?6batch_normalization_39/FusedBatchNormV3/ReadVariableOp?8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_39/ReadVariableOp?'batch_normalization_39/ReadVariableOp_1?6batch_normalization_41/FusedBatchNormV3/ReadVariableOp?8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_41/ReadVariableOp?'batch_normalization_41/ReadVariableOp_1?6batch_normalization_43/FusedBatchNormV3/ReadVariableOp?8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_43/ReadVariableOp?'batch_normalization_43/ReadVariableOp_1?6batch_normalization_45/FusedBatchNormV3/ReadVariableOp?8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_45/ReadVariableOp?'batch_normalization_45/ReadVariableOp_1?6batch_normalization_47/FusedBatchNormV3/ReadVariableOp?8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_47/ReadVariableOp?'batch_normalization_47/ReadVariableOp_1?6batch_normalization_49/FusedBatchNormV3/ReadVariableOp?8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_49/ReadVariableOp?'batch_normalization_49/ReadVariableOp_1?6batch_normalization_51/FusedBatchNormV3/ReadVariableOp?8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_51/ReadVariableOp?'batch_normalization_51/ReadVariableOp_1?6batch_normalization_53/FusedBatchNormV3/ReadVariableOp?8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_53/ReadVariableOp?'batch_normalization_53/ReadVariableOp_1? conv2d_39/BiasAdd/ReadVariableOp?conv2d_39/Conv2D/ReadVariableOp? conv2d_41/BiasAdd/ReadVariableOp?conv2d_41/Conv2D/ReadVariableOp? conv2d_43/BiasAdd/ReadVariableOp?conv2d_43/Conv2D/ReadVariableOp? conv2d_45/BiasAdd/ReadVariableOp?conv2d_45/Conv2D/ReadVariableOp? conv2d_47/BiasAdd/ReadVariableOp?conv2d_47/Conv2D/ReadVariableOp? conv2d_49/BiasAdd/ReadVariableOp?conv2d_49/Conv2D/ReadVariableOp? conv2d_51/BiasAdd/ReadVariableOp?conv2d_51/Conv2D/ReadVariableOp? conv2d_53/BiasAdd/ReadVariableOp?conv2d_53/Conv2D/ReadVariableOp? conv2d_55/BiasAdd/ReadVariableOp?conv2d_55/Conv2D/ReadVariableOp? conv2d_56/BiasAdd/ReadVariableOp?conv2d_56/Conv2D/ReadVariableOp?*conv2d_transpose_10/BiasAdd/ReadVariableOp?3conv2d_transpose_10/conv2d_transpose/ReadVariableOp?*conv2d_transpose_11/BiasAdd/ReadVariableOp?3conv2d_transpose_11/conv2d_transpose/ReadVariableOp?)conv2d_transpose_8/BiasAdd/ReadVariableOp?2conv2d_transpose_8/conv2d_transpose/ReadVariableOp?)conv2d_transpose_9/BiasAdd/ReadVariableOp?2conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
conv2d_39/Conv2D/ReadVariableOpReadVariableOp(conv2d_39_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_39/Conv2DConv2Dinputs'conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_39/BiasAdd/ReadVariableOpReadVariableOp)conv2d_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_39/BiasAddBiasAddconv2d_39/Conv2D:output:0(conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_37/ReadVariableOpReadVariableOp.batch_normalization_37_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_37/ReadVariableOp_1ReadVariableOp0batch_normalization_37_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_37/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_37_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_37_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_37/FusedBatchNormV3FusedBatchNormV3conv2d_39/BiasAdd:output:0-batch_normalization_37/ReadVariableOp:value:0/batch_normalization_37/ReadVariableOp_1:value:0>batch_normalization_37/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
activation_37/ReluRelu+batch_normalization_37/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:????????????
max_pooling2d_8/MaxPoolMaxPool activation_37/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
}
dropout_16/IdentityIdentity max_pooling2d_8/MaxPool:output:0*
T0*1
_output_shapes
:????????????
conv2d_41/Conv2D/ReadVariableOpReadVariableOp(conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_41/Conv2DConv2Ddropout_16/Identity:output:0'conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
 conv2d_41/BiasAdd/ReadVariableOpReadVariableOp)conv2d_41_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_41/BiasAddBiasAddconv2d_41/Conv2D:output:0(conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? ?
%batch_normalization_39/ReadVariableOpReadVariableOp.batch_normalization_39_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_39/ReadVariableOp_1ReadVariableOp0batch_normalization_39_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_39/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_39_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_39/FusedBatchNormV3FusedBatchNormV3conv2d_41/BiasAdd:output:0-batch_normalization_39/ReadVariableOp:value:0/batch_normalization_39/ReadVariableOp_1:value:0>batch_normalization_39/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( ?
activation_39/ReluRelu+batch_normalization_39/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? ?
max_pooling2d_9/MaxPoolMaxPool activation_39/Relu:activations:0*/
_output_shapes
:?????????@@ *
ksize
*
paddingVALID*
strides
{
dropout_17/IdentityIdentity max_pooling2d_9/MaxPool:output:0*
T0*/
_output_shapes
:?????????@@ ?
conv2d_43/Conv2D/ReadVariableOpReadVariableOp(conv2d_43_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_43/Conv2DConv2Ddropout_17/Identity:output:0'conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
 conv2d_43/BiasAdd/ReadVariableOpReadVariableOp)conv2d_43_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_43/BiasAddBiasAddconv2d_43/Conv2D:output:0(conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
%batch_normalization_41/ReadVariableOpReadVariableOp.batch_normalization_41_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_41/ReadVariableOp_1ReadVariableOp0batch_normalization_41_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_41/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_41_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_41/FusedBatchNormV3FusedBatchNormV3conv2d_43/BiasAdd:output:0-batch_normalization_41/ReadVariableOp:value:0/batch_normalization_41/ReadVariableOp_1:value:0>batch_normalization_41/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( ?
activation_41/ReluRelu+batch_normalization_41/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@@?
max_pooling2d_10/MaxPoolMaxPool activation_41/Relu:activations:0*/
_output_shapes
:?????????  @*
ksize
*
paddingVALID*
strides
|
dropout_18/IdentityIdentity!max_pooling2d_10/MaxPool:output:0*
T0*/
_output_shapes
:?????????  @?
conv2d_45/Conv2D/ReadVariableOpReadVariableOp(conv2d_45_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_45/Conv2DConv2Ddropout_18/Identity:output:0'conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
 conv2d_45/BiasAdd/ReadVariableOpReadVariableOp)conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_45/BiasAddBiasAddconv2d_45/Conv2D:output:0(conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
%batch_normalization_43/ReadVariableOpReadVariableOp.batch_normalization_43_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_43/ReadVariableOp_1ReadVariableOp0batch_normalization_43_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_43/FusedBatchNormV3FusedBatchNormV3conv2d_45/BiasAdd:output:0-batch_normalization_43/ReadVariableOp:value:0/batch_normalization_43/ReadVariableOp_1:value:0>batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( ?
activation_43/ReluRelu+batch_normalization_43/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ??
max_pooling2d_11/MaxPoolMaxPool activation_43/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
}
dropout_19/IdentityIdentity!max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:???????????
conv2d_47/Conv2D/ReadVariableOpReadVariableOp(conv2d_47_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_47/Conv2DConv2Ddropout_19/Identity:output:0'conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
 conv2d_47/BiasAdd/ReadVariableOpReadVariableOp)conv2d_47_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_47/BiasAddBiasAddconv2d_47/Conv2D:output:0(conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
%batch_normalization_45/ReadVariableOpReadVariableOp.batch_normalization_45_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_45/ReadVariableOp_1ReadVariableOp0batch_normalization_45_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_45/FusedBatchNormV3FusedBatchNormV3conv2d_47/BiasAdd:output:0-batch_normalization_45/ReadVariableOp:value:0/batch_normalization_45/ReadVariableOp_1:value:0>batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
activation_45/ReluRelu+batch_normalization_45/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????h
conv2d_transpose_8/ShapeShape activation_45/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_8/strided_sliceStridedSlice!conv2d_transpose_8/Shape:output:0/conv2d_transpose_8/strided_slice/stack:output:01conv2d_transpose_8/strided_slice/stack_1:output:01conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B : \
conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B : ]
conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
conv2d_transpose_8/stackPack)conv2d_transpose_8/strided_slice:output:0#conv2d_transpose_8/stack/1:output:0#conv2d_transpose_8/stack/2:output:0#conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_8/strided_slice_1StridedSlice!conv2d_transpose_8/stack:output:01conv2d_transpose_8/strided_slice_1/stack:output:03conv2d_transpose_8/strided_slice_1/stack_1:output:03conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_8_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
#conv2d_transpose_8/conv2d_transposeConv2DBackpropInput!conv2d_transpose_8/stack:output:0:conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0 activation_45/Relu:activations:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
)conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_transpose_8/BiasAddBiasAdd,conv2d_transpose_8/conv2d_transpose:output:01conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?[
concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_8/concatConcatV2#conv2d_transpose_8/BiasAdd:output:0 activation_43/Relu:activations:0"concatenate_8/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????  ?y
dropout_20/IdentityIdentityconcatenate_8/concat:output:0*
T0*0
_output_shapes
:?????????  ??
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_49/Conv2DConv2Ddropout_20/Identity:output:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
%batch_normalization_47/ReadVariableOpReadVariableOp.batch_normalization_47_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_47/ReadVariableOp_1ReadVariableOp0batch_normalization_47_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_47/FusedBatchNormV3FusedBatchNormV3conv2d_49/BiasAdd:output:0-batch_normalization_47/ReadVariableOp:value:0/batch_normalization_47/ReadVariableOp_1:value:0>batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( ?
activation_47/ReluRelu+batch_normalization_47/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?h
conv2d_transpose_9/ShapeShape activation_47/Relu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_transpose_9/strided_sliceStridedSlice!conv2d_transpose_9/Shape:output:0/conv2d_transpose_9/strided_slice/stack:output:01conv2d_transpose_9/strided_slice/stack_1:output:01conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@\
conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
conv2d_transpose_9/stackPack)conv2d_transpose_9/strided_slice:output:0#conv2d_transpose_9/stack/1:output:0#conv2d_transpose_9/stack/2:output:0#conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
"conv2d_transpose_9/strided_slice_1StridedSlice!conv2d_transpose_9/stack:output:01conv2d_transpose_9/strided_slice_1/stack:output:03conv2d_transpose_9/strided_slice_1/stack_1:output:03conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
2conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_9_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
#conv2d_transpose_9/conv2d_transposeConv2DBackpropInput!conv2d_transpose_9/stack:output:0:conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0 activation_47/Relu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
)conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_transpose_9/BiasAddBiasAdd,conv2d_transpose_9/conv2d_transpose:output:01conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@[
concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_9/concatConcatV2#conv2d_transpose_9/BiasAdd:output:0 activation_41/Relu:activations:0"concatenate_9/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????@@?y
dropout_21/IdentityIdentityconcatenate_9/concat:output:0*
T0*0
_output_shapes
:?????????@@??
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
conv2d_51/Conv2DConv2Ddropout_21/Identity:output:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
%batch_normalization_49/ReadVariableOpReadVariableOp.batch_normalization_49_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_49/ReadVariableOp_1ReadVariableOp0batch_normalization_49_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_49/FusedBatchNormV3FusedBatchNormV3conv2d_51/BiasAdd:output:0-batch_normalization_49/ReadVariableOp:value:0/batch_normalization_49/ReadVariableOp_1:value:0>batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( ?
activation_49/ReluRelu+batch_normalization_49/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@@i
conv2d_transpose_10/ShapeShape activation_49/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_10/strided_sliceStridedSlice"conv2d_transpose_10/Shape:output:00conv2d_transpose_10/strided_slice/stack:output:02conv2d_transpose_10/strided_slice/stack_1:output:02conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?^
conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?]
conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_transpose_10/stackPack*conv2d_transpose_10/strided_slice:output:0$conv2d_transpose_10/stack/1:output:0$conv2d_transpose_10/stack/2:output:0$conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_10/strided_slice_1StridedSlice"conv2d_transpose_10/stack:output:02conv2d_transpose_10/strided_slice_1/stack:output:04conv2d_transpose_10/strided_slice_1/stack_1:output:04conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_10_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
$conv2d_transpose_10/conv2d_transposeConv2DBackpropInput"conv2d_transpose_10/stack:output:0;conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0 activation_49/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
*conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_transpose_10/BiasAddBiasAdd-conv2d_transpose_10/conv2d_transpose:output:02conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? \
concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_10/concatConcatV2$conv2d_transpose_10/BiasAdd:output:0 activation_39/Relu:activations:0#concatenate_10/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@{
dropout_22/IdentityIdentityconcatenate_10/concat:output:0*
T0*1
_output_shapes
:???????????@?
conv2d_53/Conv2D/ReadVariableOpReadVariableOp(conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
conv2d_53/Conv2DConv2Ddropout_22/Identity:output:0'conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
 conv2d_53/BiasAdd/ReadVariableOpReadVariableOp)conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
conv2d_53/BiasAddBiasAddconv2d_53/Conv2D:output:0(conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? ?
%batch_normalization_51/ReadVariableOpReadVariableOp.batch_normalization_51_readvariableop_resource*
_output_shapes
: *
dtype0?
'batch_normalization_51/ReadVariableOp_1ReadVariableOp0batch_normalization_51_readvariableop_1_resource*
_output_shapes
: *
dtype0?
6batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
'batch_normalization_51/FusedBatchNormV3FusedBatchNormV3conv2d_53/BiasAdd:output:0-batch_normalization_51/ReadVariableOp:value:0/batch_normalization_51/ReadVariableOp_1:value:0>batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( ?
activation_51/ReluRelu+batch_normalization_51/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? i
conv2d_transpose_11/ShapeShape activation_51/Relu:activations:0*
T0*
_output_shapes
:q
'conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!conv2d_transpose_11/strided_sliceStridedSlice"conv2d_transpose_11/Shape:output:00conv2d_transpose_11/strided_slice/stack:output:02conv2d_transpose_11/strided_slice/stack_1:output:02conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?^
conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?]
conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
conv2d_transpose_11/stackPack*conv2d_transpose_11/strided_slice:output:0$conv2d_transpose_11/stack/1:output:0$conv2d_transpose_11/stack/2:output:0$conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:s
)conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#conv2d_transpose_11/strided_slice_1StridedSlice"conv2d_transpose_11/stack:output:02conv2d_transpose_11/strided_slice_1/stack:output:04conv2d_transpose_11/strided_slice_1/stack_1:output:04conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
3conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_11_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
$conv2d_transpose_11/conv2d_transposeConv2DBackpropInput"conv2d_transpose_11/stack:output:0;conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0 activation_51/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
*conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_transpose_11/BiasAddBiasAdd-conv2d_transpose_11/conv2d_transpose:output:02conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????\
concatenate_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
concatenate_11/concatConcatV2$conv2d_transpose_11/BiasAdd:output:0 activation_37/Relu:activations:0#concatenate_11/concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? {
dropout_23/IdentityIdentityconcatenate_11/concat:output:0*
T0*1
_output_shapes
:??????????? ?
conv2d_55/Conv2D/ReadVariableOpReadVariableOp(conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_55/Conv2DConv2Ddropout_23/Identity:output:0'conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
 conv2d_55/BiasAdd/ReadVariableOpReadVariableOp)conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_55/BiasAddBiasAddconv2d_55/Conv2D:output:0(conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
%batch_normalization_53/ReadVariableOpReadVariableOp.batch_normalization_53_readvariableop_resource*
_output_shapes
:*
dtype0?
'batch_normalization_53/ReadVariableOp_1ReadVariableOp0batch_normalization_53_readvariableop_1_resource*
_output_shapes
:*
dtype0?
6batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
'batch_normalization_53/FusedBatchNormV3FusedBatchNormV3conv2d_55/BiasAdd:output:0-batch_normalization_53/ReadVariableOp:value:0/batch_normalization_53/ReadVariableOp_1:value:0>batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
activation_53/ReluRelu+batch_normalization_53/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:????????????
conv2d_56/Conv2D/ReadVariableOpReadVariableOp(conv2d_56_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
conv2d_56/Conv2DConv2D activation_53/Relu:activations:0'conv2d_56/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
?
 conv2d_56/BiasAdd/ReadVariableOpReadVariableOp)conv2d_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
conv2d_56/BiasAddBiasAddconv2d_56/Conv2D:output:0(conv2d_56/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????t
conv2d_56/SigmoidSigmoidconv2d_56/BiasAdd:output:0*
T0*1
_output_shapes
:???????????n
IdentityIdentityconv2d_56/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp7^batch_normalization_37/FusedBatchNormV3/ReadVariableOp9^batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_37/ReadVariableOp(^batch_normalization_37/ReadVariableOp_17^batch_normalization_39/FusedBatchNormV3/ReadVariableOp9^batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_39/ReadVariableOp(^batch_normalization_39/ReadVariableOp_17^batch_normalization_41/FusedBatchNormV3/ReadVariableOp9^batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_41/ReadVariableOp(^batch_normalization_41/ReadVariableOp_17^batch_normalization_43/FusedBatchNormV3/ReadVariableOp9^batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_43/ReadVariableOp(^batch_normalization_43/ReadVariableOp_17^batch_normalization_45/FusedBatchNormV3/ReadVariableOp9^batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_45/ReadVariableOp(^batch_normalization_45/ReadVariableOp_17^batch_normalization_47/FusedBatchNormV3/ReadVariableOp9^batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_47/ReadVariableOp(^batch_normalization_47/ReadVariableOp_17^batch_normalization_49/FusedBatchNormV3/ReadVariableOp9^batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_49/ReadVariableOp(^batch_normalization_49/ReadVariableOp_17^batch_normalization_51/FusedBatchNormV3/ReadVariableOp9^batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_51/ReadVariableOp(^batch_normalization_51/ReadVariableOp_17^batch_normalization_53/FusedBatchNormV3/ReadVariableOp9^batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_53/ReadVariableOp(^batch_normalization_53/ReadVariableOp_1!^conv2d_39/BiasAdd/ReadVariableOp ^conv2d_39/Conv2D/ReadVariableOp!^conv2d_41/BiasAdd/ReadVariableOp ^conv2d_41/Conv2D/ReadVariableOp!^conv2d_43/BiasAdd/ReadVariableOp ^conv2d_43/Conv2D/ReadVariableOp!^conv2d_45/BiasAdd/ReadVariableOp ^conv2d_45/Conv2D/ReadVariableOp!^conv2d_47/BiasAdd/ReadVariableOp ^conv2d_47/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_51/BiasAdd/ReadVariableOp ^conv2d_51/Conv2D/ReadVariableOp!^conv2d_53/BiasAdd/ReadVariableOp ^conv2d_53/Conv2D/ReadVariableOp!^conv2d_55/BiasAdd/ReadVariableOp ^conv2d_55/Conv2D/ReadVariableOp!^conv2d_56/BiasAdd/ReadVariableOp ^conv2d_56/Conv2D/ReadVariableOp+^conv2d_transpose_10/BiasAdd/ReadVariableOp4^conv2d_transpose_10/conv2d_transpose/ReadVariableOp+^conv2d_transpose_11/BiasAdd/ReadVariableOp4^conv2d_transpose_11/conv2d_transpose/ReadVariableOp*^conv2d_transpose_8/BiasAdd/ReadVariableOp3^conv2d_transpose_8/conv2d_transpose/ReadVariableOp*^conv2d_transpose_9/BiasAdd/ReadVariableOp3^conv2d_transpose_9/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_37/FusedBatchNormV3/ReadVariableOp6batch_normalization_37/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_37/FusedBatchNormV3/ReadVariableOp_18batch_normalization_37/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_37/ReadVariableOp%batch_normalization_37/ReadVariableOp2R
'batch_normalization_37/ReadVariableOp_1'batch_normalization_37/ReadVariableOp_12p
6batch_normalization_39/FusedBatchNormV3/ReadVariableOp6batch_normalization_39/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_18batch_normalization_39/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_39/ReadVariableOp%batch_normalization_39/ReadVariableOp2R
'batch_normalization_39/ReadVariableOp_1'batch_normalization_39/ReadVariableOp_12p
6batch_normalization_41/FusedBatchNormV3/ReadVariableOp6batch_normalization_41/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_18batch_normalization_41/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_41/ReadVariableOp%batch_normalization_41/ReadVariableOp2R
'batch_normalization_41/ReadVariableOp_1'batch_normalization_41/ReadVariableOp_12p
6batch_normalization_43/FusedBatchNormV3/ReadVariableOp6batch_normalization_43/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_43/FusedBatchNormV3/ReadVariableOp_18batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_43/ReadVariableOp%batch_normalization_43/ReadVariableOp2R
'batch_normalization_43/ReadVariableOp_1'batch_normalization_43/ReadVariableOp_12p
6batch_normalization_45/FusedBatchNormV3/ReadVariableOp6batch_normalization_45/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_45/FusedBatchNormV3/ReadVariableOp_18batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_45/ReadVariableOp%batch_normalization_45/ReadVariableOp2R
'batch_normalization_45/ReadVariableOp_1'batch_normalization_45/ReadVariableOp_12p
6batch_normalization_47/FusedBatchNormV3/ReadVariableOp6batch_normalization_47/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_47/FusedBatchNormV3/ReadVariableOp_18batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_47/ReadVariableOp%batch_normalization_47/ReadVariableOp2R
'batch_normalization_47/ReadVariableOp_1'batch_normalization_47/ReadVariableOp_12p
6batch_normalization_49/FusedBatchNormV3/ReadVariableOp6batch_normalization_49/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_49/FusedBatchNormV3/ReadVariableOp_18batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_49/ReadVariableOp%batch_normalization_49/ReadVariableOp2R
'batch_normalization_49/ReadVariableOp_1'batch_normalization_49/ReadVariableOp_12p
6batch_normalization_51/FusedBatchNormV3/ReadVariableOp6batch_normalization_51/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_51/FusedBatchNormV3/ReadVariableOp_18batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_51/ReadVariableOp%batch_normalization_51/ReadVariableOp2R
'batch_normalization_51/ReadVariableOp_1'batch_normalization_51/ReadVariableOp_12p
6batch_normalization_53/FusedBatchNormV3/ReadVariableOp6batch_normalization_53/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_53/FusedBatchNormV3/ReadVariableOp_18batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_53/ReadVariableOp%batch_normalization_53/ReadVariableOp2R
'batch_normalization_53/ReadVariableOp_1'batch_normalization_53/ReadVariableOp_12D
 conv2d_39/BiasAdd/ReadVariableOp conv2d_39/BiasAdd/ReadVariableOp2B
conv2d_39/Conv2D/ReadVariableOpconv2d_39/Conv2D/ReadVariableOp2D
 conv2d_41/BiasAdd/ReadVariableOp conv2d_41/BiasAdd/ReadVariableOp2B
conv2d_41/Conv2D/ReadVariableOpconv2d_41/Conv2D/ReadVariableOp2D
 conv2d_43/BiasAdd/ReadVariableOp conv2d_43/BiasAdd/ReadVariableOp2B
conv2d_43/Conv2D/ReadVariableOpconv2d_43/Conv2D/ReadVariableOp2D
 conv2d_45/BiasAdd/ReadVariableOp conv2d_45/BiasAdd/ReadVariableOp2B
conv2d_45/Conv2D/ReadVariableOpconv2d_45/Conv2D/ReadVariableOp2D
 conv2d_47/BiasAdd/ReadVariableOp conv2d_47/BiasAdd/ReadVariableOp2B
conv2d_47/Conv2D/ReadVariableOpconv2d_47/Conv2D/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_51/BiasAdd/ReadVariableOp conv2d_51/BiasAdd/ReadVariableOp2B
conv2d_51/Conv2D/ReadVariableOpconv2d_51/Conv2D/ReadVariableOp2D
 conv2d_53/BiasAdd/ReadVariableOp conv2d_53/BiasAdd/ReadVariableOp2B
conv2d_53/Conv2D/ReadVariableOpconv2d_53/Conv2D/ReadVariableOp2D
 conv2d_55/BiasAdd/ReadVariableOp conv2d_55/BiasAdd/ReadVariableOp2B
conv2d_55/Conv2D/ReadVariableOpconv2d_55/Conv2D/ReadVariableOp2D
 conv2d_56/BiasAdd/ReadVariableOp conv2d_56/BiasAdd/ReadVariableOp2B
conv2d_56/Conv2D/ReadVariableOpconv2d_56/Conv2D/ReadVariableOp2X
*conv2d_transpose_10/BiasAdd/ReadVariableOp*conv2d_transpose_10/BiasAdd/ReadVariableOp2j
3conv2d_transpose_10/conv2d_transpose/ReadVariableOp3conv2d_transpose_10/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_11/BiasAdd/ReadVariableOp*conv2d_transpose_11/BiasAdd/ReadVariableOp2j
3conv2d_transpose_11/conv2d_transpose/ReadVariableOp3conv2d_transpose_11/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_8/BiasAdd/ReadVariableOp)conv2d_transpose_8/BiasAdd/ReadVariableOp2h
2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2conv2d_transpose_8/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_9/BiasAdd/ReadVariableOp)conv2d_transpose_9/BiasAdd/ReadVariableOp2h
2conv2d_transpose_9/conv2d_transpose/ReadVariableOp2conv2d_transpose_9/conv2d_transpose/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
H__inference_activation_37_layer_call_and_return_conditional_losses_84192

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:???????????d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80954

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_9_layer_call_fn_84886

inputs"
unknown:@?
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_80894?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
Ω
?A
 __inference__wrapped_model_80381
imgJ
0model_2_conv2d_39_conv2d_readvariableop_resource:?
1model_2_conv2d_39_biasadd_readvariableop_resource:D
6model_2_batch_normalization_37_readvariableop_resource:F
8model_2_batch_normalization_37_readvariableop_1_resource:U
Gmodel_2_batch_normalization_37_fusedbatchnormv3_readvariableop_resource:W
Imodel_2_batch_normalization_37_fusedbatchnormv3_readvariableop_1_resource:J
0model_2_conv2d_41_conv2d_readvariableop_resource: ?
1model_2_conv2d_41_biasadd_readvariableop_resource: D
6model_2_batch_normalization_39_readvariableop_resource: F
8model_2_batch_normalization_39_readvariableop_1_resource: U
Gmodel_2_batch_normalization_39_fusedbatchnormv3_readvariableop_resource: W
Imodel_2_batch_normalization_39_fusedbatchnormv3_readvariableop_1_resource: J
0model_2_conv2d_43_conv2d_readvariableop_resource: @?
1model_2_conv2d_43_biasadd_readvariableop_resource:@D
6model_2_batch_normalization_41_readvariableop_resource:@F
8model_2_batch_normalization_41_readvariableop_1_resource:@U
Gmodel_2_batch_normalization_41_fusedbatchnormv3_readvariableop_resource:@W
Imodel_2_batch_normalization_41_fusedbatchnormv3_readvariableop_1_resource:@K
0model_2_conv2d_45_conv2d_readvariableop_resource:@?@
1model_2_conv2d_45_biasadd_readvariableop_resource:	?E
6model_2_batch_normalization_43_readvariableop_resource:	?G
8model_2_batch_normalization_43_readvariableop_1_resource:	?V
Gmodel_2_batch_normalization_43_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_2_batch_normalization_43_fusedbatchnormv3_readvariableop_1_resource:	?L
0model_2_conv2d_47_conv2d_readvariableop_resource:??@
1model_2_conv2d_47_biasadd_readvariableop_resource:	?E
6model_2_batch_normalization_45_readvariableop_resource:	?G
8model_2_batch_normalization_45_readvariableop_1_resource:	?V
Gmodel_2_batch_normalization_45_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_2_batch_normalization_45_fusedbatchnormv3_readvariableop_1_resource:	?_
Cmodel_2_conv2d_transpose_8_conv2d_transpose_readvariableop_resource:??I
:model_2_conv2d_transpose_8_biasadd_readvariableop_resource:	?L
0model_2_conv2d_49_conv2d_readvariableop_resource:??@
1model_2_conv2d_49_biasadd_readvariableop_resource:	?E
6model_2_batch_normalization_47_readvariableop_resource:	?G
8model_2_batch_normalization_47_readvariableop_1_resource:	?V
Gmodel_2_batch_normalization_47_fusedbatchnormv3_readvariableop_resource:	?X
Imodel_2_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource:	?^
Cmodel_2_conv2d_transpose_9_conv2d_transpose_readvariableop_resource:@?H
:model_2_conv2d_transpose_9_biasadd_readvariableop_resource:@K
0model_2_conv2d_51_conv2d_readvariableop_resource:?@?
1model_2_conv2d_51_biasadd_readvariableop_resource:@D
6model_2_batch_normalization_49_readvariableop_resource:@F
8model_2_batch_normalization_49_readvariableop_1_resource:@U
Gmodel_2_batch_normalization_49_fusedbatchnormv3_readvariableop_resource:@W
Imodel_2_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resource:@^
Dmodel_2_conv2d_transpose_10_conv2d_transpose_readvariableop_resource: @I
;model_2_conv2d_transpose_10_biasadd_readvariableop_resource: J
0model_2_conv2d_53_conv2d_readvariableop_resource:@ ?
1model_2_conv2d_53_biasadd_readvariableop_resource: D
6model_2_batch_normalization_51_readvariableop_resource: F
8model_2_batch_normalization_51_readvariableop_1_resource: U
Gmodel_2_batch_normalization_51_fusedbatchnormv3_readvariableop_resource: W
Imodel_2_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resource: ^
Dmodel_2_conv2d_transpose_11_conv2d_transpose_readvariableop_resource: I
;model_2_conv2d_transpose_11_biasadd_readvariableop_resource:J
0model_2_conv2d_55_conv2d_readvariableop_resource: ?
1model_2_conv2d_55_biasadd_readvariableop_resource:D
6model_2_batch_normalization_53_readvariableop_resource:F
8model_2_batch_normalization_53_readvariableop_1_resource:U
Gmodel_2_batch_normalization_53_fusedbatchnormv3_readvariableop_resource:W
Imodel_2_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resource:J
0model_2_conv2d_56_conv2d_readvariableop_resource:?
1model_2_conv2d_56_biasadd_readvariableop_resource:
identity??>model_2/batch_normalization_37/FusedBatchNormV3/ReadVariableOp?@model_2/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1?-model_2/batch_normalization_37/ReadVariableOp?/model_2/batch_normalization_37/ReadVariableOp_1?>model_2/batch_normalization_39/FusedBatchNormV3/ReadVariableOp?@model_2/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1?-model_2/batch_normalization_39/ReadVariableOp?/model_2/batch_normalization_39/ReadVariableOp_1?>model_2/batch_normalization_41/FusedBatchNormV3/ReadVariableOp?@model_2/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1?-model_2/batch_normalization_41/ReadVariableOp?/model_2/batch_normalization_41/ReadVariableOp_1?>model_2/batch_normalization_43/FusedBatchNormV3/ReadVariableOp?@model_2/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1?-model_2/batch_normalization_43/ReadVariableOp?/model_2/batch_normalization_43/ReadVariableOp_1?>model_2/batch_normalization_45/FusedBatchNormV3/ReadVariableOp?@model_2/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1?-model_2/batch_normalization_45/ReadVariableOp?/model_2/batch_normalization_45/ReadVariableOp_1?>model_2/batch_normalization_47/FusedBatchNormV3/ReadVariableOp?@model_2/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1?-model_2/batch_normalization_47/ReadVariableOp?/model_2/batch_normalization_47/ReadVariableOp_1?>model_2/batch_normalization_49/FusedBatchNormV3/ReadVariableOp?@model_2/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1?-model_2/batch_normalization_49/ReadVariableOp?/model_2/batch_normalization_49/ReadVariableOp_1?>model_2/batch_normalization_51/FusedBatchNormV3/ReadVariableOp?@model_2/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1?-model_2/batch_normalization_51/ReadVariableOp?/model_2/batch_normalization_51/ReadVariableOp_1?>model_2/batch_normalization_53/FusedBatchNormV3/ReadVariableOp?@model_2/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1?-model_2/batch_normalization_53/ReadVariableOp?/model_2/batch_normalization_53/ReadVariableOp_1?(model_2/conv2d_39/BiasAdd/ReadVariableOp?'model_2/conv2d_39/Conv2D/ReadVariableOp?(model_2/conv2d_41/BiasAdd/ReadVariableOp?'model_2/conv2d_41/Conv2D/ReadVariableOp?(model_2/conv2d_43/BiasAdd/ReadVariableOp?'model_2/conv2d_43/Conv2D/ReadVariableOp?(model_2/conv2d_45/BiasAdd/ReadVariableOp?'model_2/conv2d_45/Conv2D/ReadVariableOp?(model_2/conv2d_47/BiasAdd/ReadVariableOp?'model_2/conv2d_47/Conv2D/ReadVariableOp?(model_2/conv2d_49/BiasAdd/ReadVariableOp?'model_2/conv2d_49/Conv2D/ReadVariableOp?(model_2/conv2d_51/BiasAdd/ReadVariableOp?'model_2/conv2d_51/Conv2D/ReadVariableOp?(model_2/conv2d_53/BiasAdd/ReadVariableOp?'model_2/conv2d_53/Conv2D/ReadVariableOp?(model_2/conv2d_55/BiasAdd/ReadVariableOp?'model_2/conv2d_55/Conv2D/ReadVariableOp?(model_2/conv2d_56/BiasAdd/ReadVariableOp?'model_2/conv2d_56/Conv2D/ReadVariableOp?2model_2/conv2d_transpose_10/BiasAdd/ReadVariableOp?;model_2/conv2d_transpose_10/conv2d_transpose/ReadVariableOp?2model_2/conv2d_transpose_11/BiasAdd/ReadVariableOp?;model_2/conv2d_transpose_11/conv2d_transpose/ReadVariableOp?1model_2/conv2d_transpose_8/BiasAdd/ReadVariableOp?:model_2/conv2d_transpose_8/conv2d_transpose/ReadVariableOp?1model_2/conv2d_transpose_9/BiasAdd/ReadVariableOp?:model_2/conv2d_transpose_9/conv2d_transpose/ReadVariableOp?
'model_2/conv2d_39/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_39_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_2/conv2d_39/Conv2DConv2Dimg/model_2/conv2d_39/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
(model_2/conv2d_39/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_39_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_2/conv2d_39/BiasAddBiasAdd!model_2/conv2d_39/Conv2D:output:00model_2/conv2d_39/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
-model_2/batch_normalization_37/ReadVariableOpReadVariableOp6model_2_batch_normalization_37_readvariableop_resource*
_output_shapes
:*
dtype0?
/model_2/batch_normalization_37/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_37_readvariableop_1_resource*
_output_shapes
:*
dtype0?
>model_2/batch_normalization_37/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_37_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
@model_2/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_37_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
/model_2/batch_normalization_37/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_39/BiasAdd:output:05model_2/batch_normalization_37/ReadVariableOp:value:07model_2/batch_normalization_37/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_37/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
model_2/activation_37/ReluRelu3model_2/batch_normalization_37/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:????????????
model_2/max_pooling2d_8/MaxPoolMaxPool(model_2/activation_37/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingVALID*
strides
?
model_2/dropout_16/IdentityIdentity(model_2/max_pooling2d_8/MaxPool:output:0*
T0*1
_output_shapes
:????????????
'model_2/conv2d_41/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_41_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
model_2/conv2d_41/Conv2DConv2D$model_2/dropout_16/Identity:output:0/model_2/conv2d_41/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
(model_2/conv2d_41/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_41_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model_2/conv2d_41/BiasAddBiasAdd!model_2/conv2d_41/Conv2D:output:00model_2/conv2d_41/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? ?
-model_2/batch_normalization_39/ReadVariableOpReadVariableOp6model_2_batch_normalization_39_readvariableop_resource*
_output_shapes
: *
dtype0?
/model_2/batch_normalization_39/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_39_readvariableop_1_resource*
_output_shapes
: *
dtype0?
>model_2/batch_normalization_39/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_39_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
@model_2/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_39_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
/model_2/batch_normalization_39/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_41/BiasAdd:output:05model_2/batch_normalization_39/ReadVariableOp:value:07model_2/batch_normalization_39/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_39/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( ?
model_2/activation_39/ReluRelu3model_2/batch_normalization_39/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? ?
model_2/max_pooling2d_9/MaxPoolMaxPool(model_2/activation_39/Relu:activations:0*/
_output_shapes
:?????????@@ *
ksize
*
paddingVALID*
strides
?
model_2/dropout_17/IdentityIdentity(model_2/max_pooling2d_9/MaxPool:output:0*
T0*/
_output_shapes
:?????????@@ ?
'model_2/conv2d_43/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_43_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
model_2/conv2d_43/Conv2DConv2D$model_2/dropout_17/Identity:output:0/model_2/conv2d_43/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
(model_2/conv2d_43/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_43_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model_2/conv2d_43/BiasAddBiasAdd!model_2/conv2d_43/Conv2D:output:00model_2/conv2d_43/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
-model_2/batch_normalization_41/ReadVariableOpReadVariableOp6model_2_batch_normalization_41_readvariableop_resource*
_output_shapes
:@*
dtype0?
/model_2/batch_normalization_41/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_41_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
>model_2/batch_normalization_41/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_41_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
@model_2/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_41_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
/model_2/batch_normalization_41/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_43/BiasAdd:output:05model_2/batch_normalization_41/ReadVariableOp:value:07model_2/batch_normalization_41/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_41/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( ?
model_2/activation_41/ReluRelu3model_2/batch_normalization_41/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@@?
 model_2/max_pooling2d_10/MaxPoolMaxPool(model_2/activation_41/Relu:activations:0*/
_output_shapes
:?????????  @*
ksize
*
paddingVALID*
strides
?
model_2/dropout_18/IdentityIdentity)model_2/max_pooling2d_10/MaxPool:output:0*
T0*/
_output_shapes
:?????????  @?
'model_2/conv2d_45/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_45_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
model_2/conv2d_45/Conv2DConv2D$model_2/dropout_18/Identity:output:0/model_2/conv2d_45/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
(model_2/conv2d_45/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_45_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_2/conv2d_45/BiasAddBiasAdd!model_2/conv2d_45/Conv2D:output:00model_2/conv2d_45/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
-model_2/batch_normalization_43/ReadVariableOpReadVariableOp6model_2_batch_normalization_43_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/model_2/batch_normalization_43/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_43_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
>model_2/batch_normalization_43/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_43_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
@model_2/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_43_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
/model_2/batch_normalization_43/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_45/BiasAdd:output:05model_2/batch_normalization_43/ReadVariableOp:value:07model_2/batch_normalization_43/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_43/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( ?
model_2/activation_43/ReluRelu3model_2/batch_normalization_43/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ??
 model_2/max_pooling2d_11/MaxPoolMaxPool(model_2/activation_43/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
model_2/dropout_19/IdentityIdentity)model_2/max_pooling2d_11/MaxPool:output:0*
T0*0
_output_shapes
:???????????
'model_2/conv2d_47/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_47_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
model_2/conv2d_47/Conv2DConv2D$model_2/dropout_19/Identity:output:0/model_2/conv2d_47/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
?
(model_2/conv2d_47/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_47_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_2/conv2d_47/BiasAddBiasAdd!model_2/conv2d_47/Conv2D:output:00model_2/conv2d_47/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
-model_2/batch_normalization_45/ReadVariableOpReadVariableOp6model_2_batch_normalization_45_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/model_2/batch_normalization_45/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_45_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
>model_2/batch_normalization_45/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_45_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
@model_2/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_45_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
/model_2/batch_normalization_45/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_47/BiasAdd:output:05model_2/batch_normalization_45/ReadVariableOp:value:07model_2/batch_normalization_45/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_45/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
model_2/activation_45/ReluRelu3model_2/batch_normalization_45/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????x
 model_2/conv2d_transpose_8/ShapeShape(model_2/activation_45/Relu:activations:0*
T0*
_output_shapes
:x
.model_2/conv2d_transpose_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_2/conv2d_transpose_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_2/conv2d_transpose_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(model_2/conv2d_transpose_8/strided_sliceStridedSlice)model_2/conv2d_transpose_8/Shape:output:07model_2/conv2d_transpose_8/strided_slice/stack:output:09model_2/conv2d_transpose_8/strided_slice/stack_1:output:09model_2/conv2d_transpose_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_2/conv2d_transpose_8/stack/1Const*
_output_shapes
: *
dtype0*
value	B : d
"model_2/conv2d_transpose_8/stack/2Const*
_output_shapes
: *
dtype0*
value	B : e
"model_2/conv2d_transpose_8/stack/3Const*
_output_shapes
: *
dtype0*
value
B :??
 model_2/conv2d_transpose_8/stackPack1model_2/conv2d_transpose_8/strided_slice:output:0+model_2/conv2d_transpose_8/stack/1:output:0+model_2/conv2d_transpose_8/stack/2:output:0+model_2/conv2d_transpose_8/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_2/conv2d_transpose_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_2/conv2d_transpose_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_2/conv2d_transpose_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*model_2/conv2d_transpose_8/strided_slice_1StridedSlice)model_2/conv2d_transpose_8/stack:output:09model_2/conv2d_transpose_8/strided_slice_1/stack:output:0;model_2/conv2d_transpose_8/strided_slice_1/stack_1:output:0;model_2/conv2d_transpose_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:model_2/conv2d_transpose_8/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_2_conv2d_transpose_8_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype0?
+model_2/conv2d_transpose_8/conv2d_transposeConv2DBackpropInput)model_2/conv2d_transpose_8/stack:output:0Bmodel_2/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:value:0(model_2/activation_45/Relu:activations:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
1model_2/conv2d_transpose_8/BiasAdd/ReadVariableOpReadVariableOp:model_2_conv2d_transpose_8_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
"model_2/conv2d_transpose_8/BiasAddBiasAdd4model_2/conv2d_transpose_8/conv2d_transpose:output:09model_2/conv2d_transpose_8/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?c
!model_2/concatenate_8/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model_2/concatenate_8/concatConcatV2+model_2/conv2d_transpose_8/BiasAdd:output:0(model_2/activation_43/Relu:activations:0*model_2/concatenate_8/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????  ??
model_2/dropout_20/IdentityIdentity%model_2/concatenate_8/concat:output:0*
T0*0
_output_shapes
:?????????  ??
'model_2/conv2d_49/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_49_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
model_2/conv2d_49/Conv2DConv2D$model_2/dropout_20/Identity:output:0/model_2/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
?
(model_2/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_49_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
model_2/conv2d_49/BiasAddBiasAdd!model_2/conv2d_49/Conv2D:output:00model_2/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ??
-model_2/batch_normalization_47/ReadVariableOpReadVariableOp6model_2_batch_normalization_47_readvariableop_resource*
_output_shapes	
:?*
dtype0?
/model_2/batch_normalization_47/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_47_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
>model_2/batch_normalization_47/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_47_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
@model_2/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_47_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
/model_2/batch_normalization_47/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_49/BiasAdd:output:05model_2/batch_normalization_47/ReadVariableOp:value:07model_2/batch_normalization_47/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_47/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%o?:*
is_training( ?
model_2/activation_47/ReluRelu3model_2/batch_normalization_47/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?x
 model_2/conv2d_transpose_9/ShapeShape(model_2/activation_47/Relu:activations:0*
T0*
_output_shapes
:x
.model_2/conv2d_transpose_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0model_2/conv2d_transpose_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0model_2/conv2d_transpose_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(model_2/conv2d_transpose_9/strided_sliceStridedSlice)model_2/conv2d_transpose_9/Shape:output:07model_2/conv2d_transpose_9/strided_slice/stack:output:09model_2/conv2d_transpose_9/strided_slice/stack_1:output:09model_2/conv2d_transpose_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"model_2/conv2d_transpose_9/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@d
"model_2/conv2d_transpose_9/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@d
"model_2/conv2d_transpose_9/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@?
 model_2/conv2d_transpose_9/stackPack1model_2/conv2d_transpose_9/strided_slice:output:0+model_2/conv2d_transpose_9/stack/1:output:0+model_2/conv2d_transpose_9/stack/2:output:0+model_2/conv2d_transpose_9/stack/3:output:0*
N*
T0*
_output_shapes
:z
0model_2/conv2d_transpose_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2model_2/conv2d_transpose_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2model_2/conv2d_transpose_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*model_2/conv2d_transpose_9/strided_slice_1StridedSlice)model_2/conv2d_transpose_9/stack:output:09model_2/conv2d_transpose_9/strided_slice_1/stack:output:0;model_2/conv2d_transpose_9/strided_slice_1/stack_1:output:0;model_2/conv2d_transpose_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
:model_2/conv2d_transpose_9/conv2d_transpose/ReadVariableOpReadVariableOpCmodel_2_conv2d_transpose_9_conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
+model_2/conv2d_transpose_9/conv2d_transposeConv2DBackpropInput)model_2/conv2d_transpose_9/stack:output:0Bmodel_2/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:value:0(model_2/activation_47/Relu:activations:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
1model_2/conv2d_transpose_9/BiasAdd/ReadVariableOpReadVariableOp:model_2_conv2d_transpose_9_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
"model_2/conv2d_transpose_9/BiasAddBiasAdd4model_2/conv2d_transpose_9/conv2d_transpose:output:09model_2/conv2d_transpose_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@c
!model_2/concatenate_9/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model_2/concatenate_9/concatConcatV2+model_2/conv2d_transpose_9/BiasAdd:output:0(model_2/activation_41/Relu:activations:0*model_2/concatenate_9/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????@@??
model_2/dropout_21/IdentityIdentity%model_2/concatenate_9/concat:output:0*
T0*0
_output_shapes
:?????????@@??
'model_2/conv2d_51/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_51_conv2d_readvariableop_resource*'
_output_shapes
:?@*
dtype0?
model_2/conv2d_51/Conv2DConv2D$model_2/dropout_21/Identity:output:0/model_2/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@*
paddingSAME*
strides
?
(model_2/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
model_2/conv2d_51/BiasAddBiasAdd!model_2/conv2d_51/Conv2D:output:00model_2/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@@?
-model_2/batch_normalization_49/ReadVariableOpReadVariableOp6model_2_batch_normalization_49_readvariableop_resource*
_output_shapes
:@*
dtype0?
/model_2/batch_normalization_49/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_49_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
>model_2/batch_normalization_49/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_49_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
@model_2/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_49_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
/model_2/batch_normalization_49/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_51/BiasAdd:output:05model_2/batch_normalization_49/ReadVariableOp:value:07model_2/batch_normalization_49/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_49/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@@@:@:@:@:@:*
epsilon%o?:*
is_training( ?
model_2/activation_49/ReluRelu3model_2/batch_normalization_49/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@@@y
!model_2/conv2d_transpose_10/ShapeShape(model_2/activation_49/Relu:activations:0*
T0*
_output_shapes
:y
/model_2/conv2d_transpose_10/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_2/conv2d_transpose_10/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_2/conv2d_transpose_10/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model_2/conv2d_transpose_10/strided_sliceStridedSlice*model_2/conv2d_transpose_10/Shape:output:08model_2/conv2d_transpose_10/strided_slice/stack:output:0:model_2/conv2d_transpose_10/strided_slice/stack_1:output:0:model_2/conv2d_transpose_10/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#model_2/conv2d_transpose_10/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?f
#model_2/conv2d_transpose_10/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?e
#model_2/conv2d_transpose_10/stack/3Const*
_output_shapes
: *
dtype0*
value	B : ?
!model_2/conv2d_transpose_10/stackPack2model_2/conv2d_transpose_10/strided_slice:output:0,model_2/conv2d_transpose_10/stack/1:output:0,model_2/conv2d_transpose_10/stack/2:output:0,model_2/conv2d_transpose_10/stack/3:output:0*
N*
T0*
_output_shapes
:{
1model_2/conv2d_transpose_10/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model_2/conv2d_transpose_10/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_2/conv2d_transpose_10/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model_2/conv2d_transpose_10/strided_slice_1StridedSlice*model_2/conv2d_transpose_10/stack:output:0:model_2/conv2d_transpose_10/strided_slice_1/stack:output:0<model_2/conv2d_transpose_10/strided_slice_1/stack_1:output:0<model_2/conv2d_transpose_10/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;model_2/conv2d_transpose_10/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_2_conv2d_transpose_10_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype0?
,model_2/conv2d_transpose_10/conv2d_transposeConv2DBackpropInput*model_2/conv2d_transpose_10/stack:output:0Cmodel_2/conv2d_transpose_10/conv2d_transpose/ReadVariableOp:value:0(model_2/activation_49/Relu:activations:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
2model_2/conv2d_transpose_10/BiasAdd/ReadVariableOpReadVariableOp;model_2_conv2d_transpose_10_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
#model_2/conv2d_transpose_10/BiasAddBiasAdd5model_2/conv2d_transpose_10/conv2d_transpose:output:0:model_2/conv2d_transpose_10/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? d
"model_2/concatenate_10/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model_2/concatenate_10/concatConcatV2,model_2/conv2d_transpose_10/BiasAdd:output:0(model_2/activation_39/Relu:activations:0+model_2/concatenate_10/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????@?
model_2/dropout_22/IdentityIdentity&model_2/concatenate_10/concat:output:0*
T0*1
_output_shapes
:???????????@?
'model_2/conv2d_53/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_53_conv2d_readvariableop_resource*&
_output_shapes
:@ *
dtype0?
model_2/conv2d_53/Conv2DConv2D$model_2/dropout_22/Identity:output:0/model_2/conv2d_53/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? *
paddingSAME*
strides
?
(model_2/conv2d_53/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_53_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0?
model_2/conv2d_53/BiasAddBiasAdd!model_2/conv2d_53/Conv2D:output:00model_2/conv2d_53/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:??????????? ?
-model_2/batch_normalization_51/ReadVariableOpReadVariableOp6model_2_batch_normalization_51_readvariableop_resource*
_output_shapes
: *
dtype0?
/model_2/batch_normalization_51/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_51_readvariableop_1_resource*
_output_shapes
: *
dtype0?
>model_2/batch_normalization_51/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_51_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
@model_2/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_51_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
/model_2/batch_normalization_51/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_53/BiasAdd:output:05model_2/batch_normalization_51/ReadVariableOp:value:07model_2/batch_normalization_51/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_51/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:??????????? : : : : :*
epsilon%o?:*
is_training( ?
model_2/activation_51/ReluRelu3model_2/batch_normalization_51/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:??????????? y
!model_2/conv2d_transpose_11/ShapeShape(model_2/activation_51/Relu:activations:0*
T0*
_output_shapes
:y
/model_2/conv2d_transpose_11/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1model_2/conv2d_transpose_11/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1model_2/conv2d_transpose_11/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)model_2/conv2d_transpose_11/strided_sliceStridedSlice*model_2/conv2d_transpose_11/Shape:output:08model_2/conv2d_transpose_11/strided_slice/stack:output:0:model_2/conv2d_transpose_11/strided_slice/stack_1:output:0:model_2/conv2d_transpose_11/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
#model_2/conv2d_transpose_11/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?f
#model_2/conv2d_transpose_11/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?e
#model_2/conv2d_transpose_11/stack/3Const*
_output_shapes
: *
dtype0*
value	B :?
!model_2/conv2d_transpose_11/stackPack2model_2/conv2d_transpose_11/strided_slice:output:0,model_2/conv2d_transpose_11/stack/1:output:0,model_2/conv2d_transpose_11/stack/2:output:0,model_2/conv2d_transpose_11/stack/3:output:0*
N*
T0*
_output_shapes
:{
1model_2/conv2d_transpose_11/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3model_2/conv2d_transpose_11/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3model_2/conv2d_transpose_11/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+model_2/conv2d_transpose_11/strided_slice_1StridedSlice*model_2/conv2d_transpose_11/stack:output:0:model_2/conv2d_transpose_11/strided_slice_1/stack:output:0<model_2/conv2d_transpose_11/strided_slice_1/stack_1:output:0<model_2/conv2d_transpose_11/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
;model_2/conv2d_transpose_11/conv2d_transpose/ReadVariableOpReadVariableOpDmodel_2_conv2d_transpose_11_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
,model_2/conv2d_transpose_11/conv2d_transposeConv2DBackpropInput*model_2/conv2d_transpose_11/stack:output:0Cmodel_2/conv2d_transpose_11/conv2d_transpose/ReadVariableOp:value:0(model_2/activation_51/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
2model_2/conv2d_transpose_11/BiasAdd/ReadVariableOpReadVariableOp;model_2_conv2d_transpose_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
#model_2/conv2d_transpose_11/BiasAddBiasAdd5model_2/conv2d_transpose_11/conv2d_transpose:output:0:model_2/conv2d_transpose_11/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????d
"model_2/concatenate_11/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :?
model_2/concatenate_11/concatConcatV2,model_2/conv2d_transpose_11/BiasAdd:output:0(model_2/activation_37/Relu:activations:0+model_2/concatenate_11/concat/axis:output:0*
N*
T0*1
_output_shapes
:??????????? ?
model_2/dropout_23/IdentityIdentity&model_2/concatenate_11/concat:output:0*
T0*1
_output_shapes
:??????????? ?
'model_2/conv2d_55/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_55_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
model_2/conv2d_55/Conv2DConv2D$model_2/dropout_23/Identity:output:0/model_2/conv2d_55/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
?
(model_2/conv2d_55/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_55_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_2/conv2d_55/BiasAddBiasAdd!model_2/conv2d_55/Conv2D:output:00model_2/conv2d_55/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
-model_2/batch_normalization_53/ReadVariableOpReadVariableOp6model_2_batch_normalization_53_readvariableop_resource*
_output_shapes
:*
dtype0?
/model_2/batch_normalization_53/ReadVariableOp_1ReadVariableOp8model_2_batch_normalization_53_readvariableop_1_resource*
_output_shapes
:*
dtype0?
>model_2/batch_normalization_53/FusedBatchNormV3/ReadVariableOpReadVariableOpGmodel_2_batch_normalization_53_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
@model_2/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpImodel_2_batch_normalization_53_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
/model_2/batch_normalization_53/FusedBatchNormV3FusedBatchNormV3"model_2/conv2d_55/BiasAdd:output:05model_2/batch_normalization_53/ReadVariableOp:value:07model_2/batch_normalization_53/ReadVariableOp_1:value:0Fmodel_2/batch_normalization_53/FusedBatchNormV3/ReadVariableOp:value:0Hmodel_2/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( ?
model_2/activation_53/ReluRelu3model_2/batch_normalization_53/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:????????????
'model_2/conv2d_56/Conv2D/ReadVariableOpReadVariableOp0model_2_conv2d_56_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
model_2/conv2d_56/Conv2DConv2D(model_2/activation_53/Relu:activations:0/model_2/conv2d_56/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
?
(model_2/conv2d_56/BiasAdd/ReadVariableOpReadVariableOp1model_2_conv2d_56_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
model_2/conv2d_56/BiasAddBiasAdd!model_2/conv2d_56/Conv2D:output:00model_2/conv2d_56/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:????????????
model_2/conv2d_56/SigmoidSigmoid"model_2/conv2d_56/BiasAdd:output:0*
T0*1
_output_shapes
:???????????v
IdentityIdentitymodel_2/conv2d_56/Sigmoid:y:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp?^model_2/batch_normalization_37/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_37/ReadVariableOp0^model_2/batch_normalization_37/ReadVariableOp_1?^model_2/batch_normalization_39/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_39/ReadVariableOp0^model_2/batch_normalization_39/ReadVariableOp_1?^model_2/batch_normalization_41/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_41/ReadVariableOp0^model_2/batch_normalization_41/ReadVariableOp_1?^model_2/batch_normalization_43/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_43/ReadVariableOp0^model_2/batch_normalization_43/ReadVariableOp_1?^model_2/batch_normalization_45/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_45/ReadVariableOp0^model_2/batch_normalization_45/ReadVariableOp_1?^model_2/batch_normalization_47/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_47/ReadVariableOp0^model_2/batch_normalization_47/ReadVariableOp_1?^model_2/batch_normalization_49/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_49/ReadVariableOp0^model_2/batch_normalization_49/ReadVariableOp_1?^model_2/batch_normalization_51/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_51/ReadVariableOp0^model_2/batch_normalization_51/ReadVariableOp_1?^model_2/batch_normalization_53/FusedBatchNormV3/ReadVariableOpA^model_2/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1.^model_2/batch_normalization_53/ReadVariableOp0^model_2/batch_normalization_53/ReadVariableOp_1)^model_2/conv2d_39/BiasAdd/ReadVariableOp(^model_2/conv2d_39/Conv2D/ReadVariableOp)^model_2/conv2d_41/BiasAdd/ReadVariableOp(^model_2/conv2d_41/Conv2D/ReadVariableOp)^model_2/conv2d_43/BiasAdd/ReadVariableOp(^model_2/conv2d_43/Conv2D/ReadVariableOp)^model_2/conv2d_45/BiasAdd/ReadVariableOp(^model_2/conv2d_45/Conv2D/ReadVariableOp)^model_2/conv2d_47/BiasAdd/ReadVariableOp(^model_2/conv2d_47/Conv2D/ReadVariableOp)^model_2/conv2d_49/BiasAdd/ReadVariableOp(^model_2/conv2d_49/Conv2D/ReadVariableOp)^model_2/conv2d_51/BiasAdd/ReadVariableOp(^model_2/conv2d_51/Conv2D/ReadVariableOp)^model_2/conv2d_53/BiasAdd/ReadVariableOp(^model_2/conv2d_53/Conv2D/ReadVariableOp)^model_2/conv2d_55/BiasAdd/ReadVariableOp(^model_2/conv2d_55/Conv2D/ReadVariableOp)^model_2/conv2d_56/BiasAdd/ReadVariableOp(^model_2/conv2d_56/Conv2D/ReadVariableOp3^model_2/conv2d_transpose_10/BiasAdd/ReadVariableOp<^model_2/conv2d_transpose_10/conv2d_transpose/ReadVariableOp3^model_2/conv2d_transpose_11/BiasAdd/ReadVariableOp<^model_2/conv2d_transpose_11/conv2d_transpose/ReadVariableOp2^model_2/conv2d_transpose_8/BiasAdd/ReadVariableOp;^model_2/conv2d_transpose_8/conv2d_transpose/ReadVariableOp2^model_2/conv2d_transpose_9/BiasAdd/ReadVariableOp;^model_2/conv2d_transpose_9/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2?
>model_2/batch_normalization_37/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_37/FusedBatchNormV3/ReadVariableOp2?
@model_2/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_37/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_37/ReadVariableOp-model_2/batch_normalization_37/ReadVariableOp2b
/model_2/batch_normalization_37/ReadVariableOp_1/model_2/batch_normalization_37/ReadVariableOp_12?
>model_2/batch_normalization_39/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_39/FusedBatchNormV3/ReadVariableOp2?
@model_2/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_39/ReadVariableOp-model_2/batch_normalization_39/ReadVariableOp2b
/model_2/batch_normalization_39/ReadVariableOp_1/model_2/batch_normalization_39/ReadVariableOp_12?
>model_2/batch_normalization_41/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_41/FusedBatchNormV3/ReadVariableOp2?
@model_2/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_41/ReadVariableOp-model_2/batch_normalization_41/ReadVariableOp2b
/model_2/batch_normalization_41/ReadVariableOp_1/model_2/batch_normalization_41/ReadVariableOp_12?
>model_2/batch_normalization_43/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_43/FusedBatchNormV3/ReadVariableOp2?
@model_2/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_43/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_43/ReadVariableOp-model_2/batch_normalization_43/ReadVariableOp2b
/model_2/batch_normalization_43/ReadVariableOp_1/model_2/batch_normalization_43/ReadVariableOp_12?
>model_2/batch_normalization_45/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_45/FusedBatchNormV3/ReadVariableOp2?
@model_2/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_45/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_45/ReadVariableOp-model_2/batch_normalization_45/ReadVariableOp2b
/model_2/batch_normalization_45/ReadVariableOp_1/model_2/batch_normalization_45/ReadVariableOp_12?
>model_2/batch_normalization_47/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_47/FusedBatchNormV3/ReadVariableOp2?
@model_2/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_47/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_47/ReadVariableOp-model_2/batch_normalization_47/ReadVariableOp2b
/model_2/batch_normalization_47/ReadVariableOp_1/model_2/batch_normalization_47/ReadVariableOp_12?
>model_2/batch_normalization_49/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_49/FusedBatchNormV3/ReadVariableOp2?
@model_2/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_49/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_49/ReadVariableOp-model_2/batch_normalization_49/ReadVariableOp2b
/model_2/batch_normalization_49/ReadVariableOp_1/model_2/batch_normalization_49/ReadVariableOp_12?
>model_2/batch_normalization_51/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_51/FusedBatchNormV3/ReadVariableOp2?
@model_2/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_51/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_51/ReadVariableOp-model_2/batch_normalization_51/ReadVariableOp2b
/model_2/batch_normalization_51/ReadVariableOp_1/model_2/batch_normalization_51/ReadVariableOp_12?
>model_2/batch_normalization_53/FusedBatchNormV3/ReadVariableOp>model_2/batch_normalization_53/FusedBatchNormV3/ReadVariableOp2?
@model_2/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_1@model_2/batch_normalization_53/FusedBatchNormV3/ReadVariableOp_12^
-model_2/batch_normalization_53/ReadVariableOp-model_2/batch_normalization_53/ReadVariableOp2b
/model_2/batch_normalization_53/ReadVariableOp_1/model_2/batch_normalization_53/ReadVariableOp_12T
(model_2/conv2d_39/BiasAdd/ReadVariableOp(model_2/conv2d_39/BiasAdd/ReadVariableOp2R
'model_2/conv2d_39/Conv2D/ReadVariableOp'model_2/conv2d_39/Conv2D/ReadVariableOp2T
(model_2/conv2d_41/BiasAdd/ReadVariableOp(model_2/conv2d_41/BiasAdd/ReadVariableOp2R
'model_2/conv2d_41/Conv2D/ReadVariableOp'model_2/conv2d_41/Conv2D/ReadVariableOp2T
(model_2/conv2d_43/BiasAdd/ReadVariableOp(model_2/conv2d_43/BiasAdd/ReadVariableOp2R
'model_2/conv2d_43/Conv2D/ReadVariableOp'model_2/conv2d_43/Conv2D/ReadVariableOp2T
(model_2/conv2d_45/BiasAdd/ReadVariableOp(model_2/conv2d_45/BiasAdd/ReadVariableOp2R
'model_2/conv2d_45/Conv2D/ReadVariableOp'model_2/conv2d_45/Conv2D/ReadVariableOp2T
(model_2/conv2d_47/BiasAdd/ReadVariableOp(model_2/conv2d_47/BiasAdd/ReadVariableOp2R
'model_2/conv2d_47/Conv2D/ReadVariableOp'model_2/conv2d_47/Conv2D/ReadVariableOp2T
(model_2/conv2d_49/BiasAdd/ReadVariableOp(model_2/conv2d_49/BiasAdd/ReadVariableOp2R
'model_2/conv2d_49/Conv2D/ReadVariableOp'model_2/conv2d_49/Conv2D/ReadVariableOp2T
(model_2/conv2d_51/BiasAdd/ReadVariableOp(model_2/conv2d_51/BiasAdd/ReadVariableOp2R
'model_2/conv2d_51/Conv2D/ReadVariableOp'model_2/conv2d_51/Conv2D/ReadVariableOp2T
(model_2/conv2d_53/BiasAdd/ReadVariableOp(model_2/conv2d_53/BiasAdd/ReadVariableOp2R
'model_2/conv2d_53/Conv2D/ReadVariableOp'model_2/conv2d_53/Conv2D/ReadVariableOp2T
(model_2/conv2d_55/BiasAdd/ReadVariableOp(model_2/conv2d_55/BiasAdd/ReadVariableOp2R
'model_2/conv2d_55/Conv2D/ReadVariableOp'model_2/conv2d_55/Conv2D/ReadVariableOp2T
(model_2/conv2d_56/BiasAdd/ReadVariableOp(model_2/conv2d_56/BiasAdd/ReadVariableOp2R
'model_2/conv2d_56/Conv2D/ReadVariableOp'model_2/conv2d_56/Conv2D/ReadVariableOp2h
2model_2/conv2d_transpose_10/BiasAdd/ReadVariableOp2model_2/conv2d_transpose_10/BiasAdd/ReadVariableOp2z
;model_2/conv2d_transpose_10/conv2d_transpose/ReadVariableOp;model_2/conv2d_transpose_10/conv2d_transpose/ReadVariableOp2h
2model_2/conv2d_transpose_11/BiasAdd/ReadVariableOp2model_2/conv2d_transpose_11/BiasAdd/ReadVariableOp2z
;model_2/conv2d_transpose_11/conv2d_transpose/ReadVariableOp;model_2/conv2d_transpose_11/conv2d_transpose/ReadVariableOp2f
1model_2/conv2d_transpose_8/BiasAdd/ReadVariableOp1model_2/conv2d_transpose_8/BiasAdd/ReadVariableOp2x
:model_2/conv2d_transpose_8/conv2d_transpose/ReadVariableOp:model_2/conv2d_transpose_8/conv2d_transpose/ReadVariableOp2f
1model_2/conv2d_transpose_9/BiasAdd/ReadVariableOp1model_2/conv2d_transpose_9/BiasAdd/ReadVariableOp2x
:model_2/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:model_2/conv2d_transpose_9/conv2d_transpose/ReadVariableOp:V R
1
_output_shapes
:???????????

_user_specified_nameimg
?
?
'__inference_model_2_layer_call_fn_83320

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5: 
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@%

unknown_17:@?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:	?

unknown_22:	?&

unknown_23:??

unknown_24:	?

unknown_25:	?

unknown_26:	?

unknown_27:	?

unknown_28:	?&

unknown_29:??

unknown_30:	?&

unknown_31:??

unknown_32:	?

unknown_33:	?

unknown_34:	?

unknown_35:	?

unknown_36:	?%

unknown_37:@?

unknown_38:@%

unknown_39:?@

unknown_40:@

unknown_41:@

unknown_42:@

unknown_43:@

unknown_44:@$

unknown_45: @

unknown_46: $

unknown_47:@ 

unknown_48: 

unknown_49: 

unknown_50: 

unknown_51: 

unknown_52: $

unknown_53: 

unknown_54:$

unknown_55: 

unknown_56:

unknown_57:

unknown_58:

unknown_59:

unknown_60:$

unknown_61:

unknown_62:
identity??StatefulPartitionedCall?	
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54
unknown_55
unknown_56
unknown_57
unknown_58
unknown_59
unknown_60
unknown_61
unknown_62*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*b
_read_only_resource_inputsD
B@	
 !"#$%&'()*+,-./0123456789:;<=>?@*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_model_2_layer_call_and_return_conditional_losses_81610y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
D__inference_conv2d_39_layer_call_and_return_conditional_losses_81198

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
H__inference_activation_37_layer_call_and_return_conditional_losses_81218

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:???????????d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
B__inference_model_2_layer_call_and_return_conditional_losses_82866
img)
conv2d_39_82689:
conv2d_39_82691:*
batch_normalization_37_82694:*
batch_normalization_37_82696:*
batch_normalization_37_82698:*
batch_normalization_37_82700:)
conv2d_41_82706: 
conv2d_41_82708: *
batch_normalization_39_82711: *
batch_normalization_39_82713: *
batch_normalization_39_82715: *
batch_normalization_39_82717: )
conv2d_43_82723: @
conv2d_43_82725:@*
batch_normalization_41_82728:@*
batch_normalization_41_82730:@*
batch_normalization_41_82732:@*
batch_normalization_41_82734:@*
conv2d_45_82740:@?
conv2d_45_82742:	?+
batch_normalization_43_82745:	?+
batch_normalization_43_82747:	?+
batch_normalization_43_82749:	?+
batch_normalization_43_82751:	?+
conv2d_47_82757:??
conv2d_47_82759:	?+
batch_normalization_45_82762:	?+
batch_normalization_45_82764:	?+
batch_normalization_45_82766:	?+
batch_normalization_45_82768:	?4
conv2d_transpose_8_82772:??'
conv2d_transpose_8_82774:	?+
conv2d_49_82779:??
conv2d_49_82781:	?+
batch_normalization_47_82784:	?+
batch_normalization_47_82786:	?+
batch_normalization_47_82788:	?+
batch_normalization_47_82790:	?3
conv2d_transpose_9_82794:@?&
conv2d_transpose_9_82796:@*
conv2d_51_82801:?@
conv2d_51_82803:@*
batch_normalization_49_82806:@*
batch_normalization_49_82808:@*
batch_normalization_49_82810:@*
batch_normalization_49_82812:@3
conv2d_transpose_10_82816: @'
conv2d_transpose_10_82818: )
conv2d_53_82823:@ 
conv2d_53_82825: *
batch_normalization_51_82828: *
batch_normalization_51_82830: *
batch_normalization_51_82832: *
batch_normalization_51_82834: 3
conv2d_transpose_11_82838: '
conv2d_transpose_11_82840:)
conv2d_55_82845: 
conv2d_55_82847:*
batch_normalization_53_82850:*
batch_normalization_53_82852:*
batch_normalization_53_82854:*
batch_normalization_53_82856:)
conv2d_56_82860:
conv2d_56_82862:
identity??.batch_normalization_37/StatefulPartitionedCall?.batch_normalization_39/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?.batch_normalization_53/StatefulPartitionedCall?!conv2d_39/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall?!conv2d_45/StatefulPartitionedCall?!conv2d_47/StatefulPartitionedCall?!conv2d_49/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?!conv2d_53/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall?!conv2d_56/StatefulPartitionedCall?+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCallimgconv2d_39_82689conv2d_39_82691*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_81198?
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0batch_normalization_37_82694batch_normalization_37_82696batch_normalization_37_82698batch_normalization_37_82700*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_80403?
activation_37/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_37_layer_call_and_return_conditional_losses_81218?
max_pooling2d_8/PartitionedCallPartitionedCall&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_80454?
dropout_16/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_81226?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0conv2d_41_82706conv2d_41_82708*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_81238?
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0batch_normalization_39_82711batch_normalization_39_82713batch_normalization_39_82715batch_normalization_39_82717*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_80479?
activation_39/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_39_layer_call_and_return_conditional_losses_81258?
max_pooling2d_9/PartitionedCallPartitionedCall&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_80530?
dropout_17/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_81266?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0conv2d_43_82723conv2d_43_82725*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_81278?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0batch_normalization_41_82728batch_normalization_41_82730batch_normalization_41_82732batch_normalization_41_82734*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_80555?
activation_41/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_81298?
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_80606?
dropout_18/PartitionedCallPartitionedCall)max_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_81306?
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0conv2d_45_82740conv2d_45_82742*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_81318?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0batch_normalization_43_82745batch_normalization_43_82747batch_normalization_43_82749batch_normalization_43_82751*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_80631?
activation_43/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_43_layer_call_and_return_conditional_losses_81338?
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_80682?
dropout_19/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_81346?
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0conv2d_47_82757conv2d_47_82759*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_81358?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0batch_normalization_45_82762batch_normalization_45_82764batch_normalization_45_82766batch_normalization_45_82768*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80707?
activation_45/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_45_layer_call_and_return_conditional_losses_81378?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall&activation_45/PartitionedCall:output:0conv2d_transpose_8_82772conv2d_transpose_8_82774*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_80786?
concatenate_8/PartitionedCallPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0&activation_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_8_layer_call_and_return_conditional_losses_81392?
dropout_20/PartitionedCallPartitionedCall&concatenate_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_81399?
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0conv2d_49_82779conv2d_49_82781*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_81411?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0batch_normalization_47_82784batch_normalization_47_82786batch_normalization_47_82788batch_normalization_47_82790*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80815?
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_47_layer_call_and_return_conditional_losses_81431?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall&activation_47/PartitionedCall:output:0conv2d_transpose_9_82794conv2d_transpose_9_82796*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_80894?
concatenate_9/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0&activation_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_9_layer_call_and_return_conditional_losses_81445?
dropout_21/PartitionedCallPartitionedCall&concatenate_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_81452?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0conv2d_51_82801conv2d_51_82803*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_51_layer_call_and_return_conditional_losses_81464?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0batch_normalization_49_82806batch_normalization_49_82808batch_normalization_49_82810batch_normalization_49_82812*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80923?
activation_49/PartitionedCallPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_49_layer_call_and_return_conditional_losses_81484?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall&activation_49/PartitionedCall:output:0conv2d_transpose_10_82816conv2d_transpose_10_82818*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_81002?
concatenate_10/PartitionedCallPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_10_layer_call_and_return_conditional_losses_81498?
dropout_22/PartitionedCallPartitionedCall'concatenate_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_22_layer_call_and_return_conditional_losses_81505?
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0conv2d_53_82823conv2d_53_82825*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_53_layer_call_and_return_conditional_losses_81517?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0batch_normalization_51_82828batch_normalization_51_82830batch_normalization_51_82832batch_normalization_51_82834*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_81031?
activation_51/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_51_layer_call_and_return_conditional_losses_81537?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall&activation_51/PartitionedCall:output:0conv2d_transpose_11_82838conv2d_transpose_11_82840*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_81110?
concatenate_11/PartitionedCallPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_11_layer_call_and_return_conditional_losses_81551?
dropout_23/PartitionedCallPartitionedCall'concatenate_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_23_layer_call_and_return_conditional_losses_81558?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0conv2d_55_82845conv2d_55_82847*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_55_layer_call_and_return_conditional_losses_81570?
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_53_82850batch_normalization_53_82852batch_normalization_53_82854batch_normalization_53_82856*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_81139?
activation_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_53_layer_call_and_return_conditional_losses_81590?
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall&activation_53/PartitionedCall:output:0conv2d_56_82860conv2d_56_82862*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_56_layer_call_and_return_conditional_losses_81603?
IdentityIdentity*conv2d_56/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall"^conv2d_56/StatefulPartitionedCall,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall:V R
1
_output_shapes
:???????????

_user_specified_nameimg
?
c
E__inference_dropout_18_layer_call_and_return_conditional_losses_81306

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????  @c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????  @"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  @:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?
c
E__inference_dropout_20_layer_call_and_return_conditional_losses_84774

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????  ?d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????  ?"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

d
E__inference_dropout_20_layer_call_and_return_conditional_losses_84786

inputs
identity?R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??m
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????  ?C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????  ?*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????  ?x
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????  ?r
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????  ?b
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????  ?"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
? 
?
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_84919

inputsC
(conv2d_transpose_readvariableop_resource:@?-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
B__inference_model_2_layer_call_and_return_conditional_losses_81610

inputs)
conv2d_39_81199:
conv2d_39_81201:*
batch_normalization_37_81204:*
batch_normalization_37_81206:*
batch_normalization_37_81208:*
batch_normalization_37_81210:)
conv2d_41_81239: 
conv2d_41_81241: *
batch_normalization_39_81244: *
batch_normalization_39_81246: *
batch_normalization_39_81248: *
batch_normalization_39_81250: )
conv2d_43_81279: @
conv2d_43_81281:@*
batch_normalization_41_81284:@*
batch_normalization_41_81286:@*
batch_normalization_41_81288:@*
batch_normalization_41_81290:@*
conv2d_45_81319:@?
conv2d_45_81321:	?+
batch_normalization_43_81324:	?+
batch_normalization_43_81326:	?+
batch_normalization_43_81328:	?+
batch_normalization_43_81330:	?+
conv2d_47_81359:??
conv2d_47_81361:	?+
batch_normalization_45_81364:	?+
batch_normalization_45_81366:	?+
batch_normalization_45_81368:	?+
batch_normalization_45_81370:	?4
conv2d_transpose_8_81380:??'
conv2d_transpose_8_81382:	?+
conv2d_49_81412:??
conv2d_49_81414:	?+
batch_normalization_47_81417:	?+
batch_normalization_47_81419:	?+
batch_normalization_47_81421:	?+
batch_normalization_47_81423:	?3
conv2d_transpose_9_81433:@?&
conv2d_transpose_9_81435:@*
conv2d_51_81465:?@
conv2d_51_81467:@*
batch_normalization_49_81470:@*
batch_normalization_49_81472:@*
batch_normalization_49_81474:@*
batch_normalization_49_81476:@3
conv2d_transpose_10_81486: @'
conv2d_transpose_10_81488: )
conv2d_53_81518:@ 
conv2d_53_81520: *
batch_normalization_51_81523: *
batch_normalization_51_81525: *
batch_normalization_51_81527: *
batch_normalization_51_81529: 3
conv2d_transpose_11_81539: '
conv2d_transpose_11_81541:)
conv2d_55_81571: 
conv2d_55_81573:*
batch_normalization_53_81576:*
batch_normalization_53_81578:*
batch_normalization_53_81580:*
batch_normalization_53_81582:)
conv2d_56_81604:
conv2d_56_81606:
identity??.batch_normalization_37/StatefulPartitionedCall?.batch_normalization_39/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?.batch_normalization_43/StatefulPartitionedCall?.batch_normalization_45/StatefulPartitionedCall?.batch_normalization_47/StatefulPartitionedCall?.batch_normalization_49/StatefulPartitionedCall?.batch_normalization_51/StatefulPartitionedCall?.batch_normalization_53/StatefulPartitionedCall?!conv2d_39/StatefulPartitionedCall?!conv2d_41/StatefulPartitionedCall?!conv2d_43/StatefulPartitionedCall?!conv2d_45/StatefulPartitionedCall?!conv2d_47/StatefulPartitionedCall?!conv2d_49/StatefulPartitionedCall?!conv2d_51/StatefulPartitionedCall?!conv2d_53/StatefulPartitionedCall?!conv2d_55/StatefulPartitionedCall?!conv2d_56/StatefulPartitionedCall?+conv2d_transpose_10/StatefulPartitionedCall?+conv2d_transpose_11/StatefulPartitionedCall?*conv2d_transpose_8/StatefulPartitionedCall?*conv2d_transpose_9/StatefulPartitionedCall?
!conv2d_39/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_39_81199conv2d_39_81201*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_39_layer_call_and_return_conditional_losses_81198?
.batch_normalization_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_39/StatefulPartitionedCall:output:0batch_normalization_37_81204batch_normalization_37_81206batch_normalization_37_81208batch_normalization_37_81210*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_80403?
activation_37/PartitionedCallPartitionedCall7batch_normalization_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_37_layer_call_and_return_conditional_losses_81218?
max_pooling2d_8/PartitionedCallPartitionedCall&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_80454?
dropout_16/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_16_layer_call_and_return_conditional_losses_81226?
!conv2d_41/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0conv2d_41_81239conv2d_41_81241*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_41_layer_call_and_return_conditional_losses_81238?
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall*conv2d_41/StatefulPartitionedCall:output:0batch_normalization_39_81244batch_normalization_39_81246batch_normalization_39_81248batch_normalization_39_81250*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_80479?
activation_39/PartitionedCallPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_39_layer_call_and_return_conditional_losses_81258?
max_pooling2d_9/PartitionedCallPartitionedCall&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_80530?
dropout_17/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_17_layer_call_and_return_conditional_losses_81266?
!conv2d_43/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0conv2d_43_81279conv2d_43_81281*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_81278?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall*conv2d_43/StatefulPartitionedCall:output:0batch_normalization_41_81284batch_normalization_41_81286batch_normalization_41_81288batch_normalization_41_81290*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_80555?
activation_41/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_81298?
 max_pooling2d_10/PartitionedCallPartitionedCall&activation_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_80606?
dropout_18/PartitionedCallPartitionedCall)max_pooling2d_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_18_layer_call_and_return_conditional_losses_81306?
!conv2d_45/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0conv2d_45_81319conv2d_45_81321*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_81318?
.batch_normalization_43/StatefulPartitionedCallStatefulPartitionedCall*conv2d_45/StatefulPartitionedCall:output:0batch_normalization_43_81324batch_normalization_43_81326batch_normalization_43_81328batch_normalization_43_81330*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_80631?
activation_43/PartitionedCallPartitionedCall7batch_normalization_43/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_43_layer_call_and_return_conditional_losses_81338?
 max_pooling2d_11/PartitionedCallPartitionedCall&activation_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_80682?
dropout_19/PartitionedCallPartitionedCall)max_pooling2d_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_19_layer_call_and_return_conditional_losses_81346?
!conv2d_47/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0conv2d_47_81359conv2d_47_81361*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_47_layer_call_and_return_conditional_losses_81358?
.batch_normalization_45/StatefulPartitionedCallStatefulPartitionedCall*conv2d_47/StatefulPartitionedCall:output:0batch_normalization_45_81364batch_normalization_45_81366batch_normalization_45_81368batch_normalization_45_81370*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80707?
activation_45/PartitionedCallPartitionedCall7batch_normalization_45/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_45_layer_call_and_return_conditional_losses_81378?
*conv2d_transpose_8/StatefulPartitionedCallStatefulPartitionedCall&activation_45/PartitionedCall:output:0conv2d_transpose_8_81380conv2d_transpose_8_81382*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_80786?
concatenate_8/PartitionedCallPartitionedCall3conv2d_transpose_8/StatefulPartitionedCall:output:0&activation_43/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_8_layer_call_and_return_conditional_losses_81392?
dropout_20/PartitionedCallPartitionedCall&concatenate_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_20_layer_call_and_return_conditional_losses_81399?
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0conv2d_49_81412conv2d_49_81414*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_81411?
.batch_normalization_47/StatefulPartitionedCallStatefulPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0batch_normalization_47_81417batch_normalization_47_81419batch_normalization_47_81421batch_normalization_47_81423*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_80815?
activation_47/PartitionedCallPartitionedCall7batch_normalization_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_47_layer_call_and_return_conditional_losses_81431?
*conv2d_transpose_9/StatefulPartitionedCallStatefulPartitionedCall&activation_47/PartitionedCall:output:0conv2d_transpose_9_81433conv2d_transpose_9_81435*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_80894?
concatenate_9/PartitionedCallPartitionedCall3conv2d_transpose_9/StatefulPartitionedCall:output:0&activation_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_concatenate_9_layer_call_and_return_conditional_losses_81445?
dropout_21/PartitionedCallPartitionedCall&concatenate_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_21_layer_call_and_return_conditional_losses_81452?
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0conv2d_51_81465conv2d_51_81467*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_51_layer_call_and_return_conditional_losses_81464?
.batch_normalization_49/StatefulPartitionedCallStatefulPartitionedCall*conv2d_51/StatefulPartitionedCall:output:0batch_normalization_49_81470batch_normalization_49_81472batch_normalization_49_81474batch_normalization_49_81476*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_80923?
activation_49/PartitionedCallPartitionedCall7batch_normalization_49/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_49_layer_call_and_return_conditional_losses_81484?
+conv2d_transpose_10/StatefulPartitionedCallStatefulPartitionedCall&activation_49/PartitionedCall:output:0conv2d_transpose_10_81486conv2d_transpose_10_81488*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_81002?
concatenate_10/PartitionedCallPartitionedCall4conv2d_transpose_10/StatefulPartitionedCall:output:0&activation_39/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_10_layer_call_and_return_conditional_losses_81498?
dropout_22/PartitionedCallPartitionedCall'concatenate_10/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_22_layer_call_and_return_conditional_losses_81505?
!conv2d_53/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0conv2d_53_81518conv2d_53_81520*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_53_layer_call_and_return_conditional_losses_81517?
.batch_normalization_51/StatefulPartitionedCallStatefulPartitionedCall*conv2d_53/StatefulPartitionedCall:output:0batch_normalization_51_81523batch_normalization_51_81525batch_normalization_51_81527batch_normalization_51_81529*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_81031?
activation_51/PartitionedCallPartitionedCall7batch_normalization_51/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_51_layer_call_and_return_conditional_losses_81537?
+conv2d_transpose_11/StatefulPartitionedCallStatefulPartitionedCall&activation_51/PartitionedCall:output:0conv2d_transpose_11_81539conv2d_transpose_11_81541*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_81110?
concatenate_11/PartitionedCallPartitionedCall4conv2d_transpose_11/StatefulPartitionedCall:output:0&activation_37/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_concatenate_11_layer_call_and_return_conditional_losses_81551?
dropout_23/PartitionedCallPartitionedCall'concatenate_11/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:??????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dropout_23_layer_call_and_return_conditional_losses_81558?
!conv2d_55/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0conv2d_55_81571conv2d_55_81573*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_55_layer_call_and_return_conditional_losses_81570?
.batch_normalization_53/StatefulPartitionedCallStatefulPartitionedCall*conv2d_55/StatefulPartitionedCall:output:0batch_normalization_53_81576batch_normalization_53_81578batch_normalization_53_81580batch_normalization_53_81582*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_81139?
activation_53/PartitionedCallPartitionedCall7batch_normalization_53/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_53_layer_call_and_return_conditional_losses_81590?
!conv2d_56/StatefulPartitionedCallStatefulPartitionedCall&activation_53/PartitionedCall:output:0conv2d_56_81604conv2d_56_81606*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_56_layer_call_and_return_conditional_losses_81603?
IdentityIdentity*conv2d_56/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:????????????
NoOpNoOp/^batch_normalization_37/StatefulPartitionedCall/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall/^batch_normalization_43/StatefulPartitionedCall/^batch_normalization_45/StatefulPartitionedCall/^batch_normalization_47/StatefulPartitionedCall/^batch_normalization_49/StatefulPartitionedCall/^batch_normalization_51/StatefulPartitionedCall/^batch_normalization_53/StatefulPartitionedCall"^conv2d_39/StatefulPartitionedCall"^conv2d_41/StatefulPartitionedCall"^conv2d_43/StatefulPartitionedCall"^conv2d_45/StatefulPartitionedCall"^conv2d_47/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall"^conv2d_53/StatefulPartitionedCall"^conv2d_55/StatefulPartitionedCall"^conv2d_56/StatefulPartitionedCall,^conv2d_transpose_10/StatefulPartitionedCall,^conv2d_transpose_11/StatefulPartitionedCall+^conv2d_transpose_8/StatefulPartitionedCall+^conv2d_transpose_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_37/StatefulPartitionedCall.batch_normalization_37/StatefulPartitionedCall2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2`
.batch_normalization_43/StatefulPartitionedCall.batch_normalization_43/StatefulPartitionedCall2`
.batch_normalization_45/StatefulPartitionedCall.batch_normalization_45/StatefulPartitionedCall2`
.batch_normalization_47/StatefulPartitionedCall.batch_normalization_47/StatefulPartitionedCall2`
.batch_normalization_49/StatefulPartitionedCall.batch_normalization_49/StatefulPartitionedCall2`
.batch_normalization_51/StatefulPartitionedCall.batch_normalization_51/StatefulPartitionedCall2`
.batch_normalization_53/StatefulPartitionedCall.batch_normalization_53/StatefulPartitionedCall2F
!conv2d_39/StatefulPartitionedCall!conv2d_39/StatefulPartitionedCall2F
!conv2d_41/StatefulPartitionedCall!conv2d_41/StatefulPartitionedCall2F
!conv2d_43/StatefulPartitionedCall!conv2d_43/StatefulPartitionedCall2F
!conv2d_45/StatefulPartitionedCall!conv2d_45/StatefulPartitionedCall2F
!conv2d_47/StatefulPartitionedCall!conv2d_47/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall2F
!conv2d_53/StatefulPartitionedCall!conv2d_53/StatefulPartitionedCall2F
!conv2d_55/StatefulPartitionedCall!conv2d_55/StatefulPartitionedCall2F
!conv2d_56/StatefulPartitionedCall!conv2d_56/StatefulPartitionedCall2Z
+conv2d_transpose_10/StatefulPartitionedCall+conv2d_transpose_10/StatefulPartitionedCall2Z
+conv2d_transpose_11/StatefulPartitionedCall+conv2d_transpose_11/StatefulPartitionedCall2X
*conv2d_transpose_8/StatefulPartitionedCall*conv2d_transpose_8/StatefulPartitionedCall2X
*conv2d_transpose_9/StatefulPartitionedCall*conv2d_transpose_9/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
d
H__inference_activation_49_layer_call_and_return_conditional_losses_81484

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@@@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
c
E__inference_dropout_16_layer_call_and_return_conditional_losses_84217

inputs

identity_1X
IdentityIdentityinputs*
T0*1
_output_shapes
:???????????e

Identity_1IdentityIdentity:output:0*
T0*1
_output_shapes
:???????????"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_80707

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_21_layer_call_and_return_conditional_losses_81452

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????@@?d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????@@?"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
I
-__inference_activation_49_layer_call_fn_85045

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_49_layer_call_and_return_conditional_losses_81484h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
d
H__inference_activation_39_layer_call_and_return_conditional_losses_81258

inputs
identityP
ReluReluinputs*
T0*1
_output_shapes
:??????????? d
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:??????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:??????????? :Y U
1
_output_shapes
:??????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_84438

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
)__inference_conv2d_43_layer_call_fn_84366

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_43_layer_call_and_return_conditional_losses_81278w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@@@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@ 
 
_user_specified_nameinputs
?
I
-__inference_activation_41_layer_call_fn_84443

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_activation_41_layer_call_and_return_conditional_losses_81298h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_84694

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
? 
?
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_81110

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????y
IdentityIdentityBiasAdd:output:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
)__inference_conv2d_45_layer_call_fn_84494

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????  ?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_45_layer_call_and_return_conditional_losses_81318x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  @
 
_user_specified_nameinputs
?	
?
6__inference_batch_normalization_39_layer_call_fn_84274

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_80510?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_84164

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
c
E__inference_dropout_20_layer_call_and_return_conditional_losses_81399

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????  ?d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????  ?"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_80631

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_10_layer_call_fn_85059

inputs!
unknown: @
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_81002?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
d
H__inference_activation_41_layer_call_and_return_conditional_losses_84448

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@@@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@@:W S
/
_output_shapes
:?????????@@@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
=
img6
serving_default_img:0???????????G
	conv2d_56:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??	
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer_with_weights-3
layer-7
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer_with_weights-7
layer-17
layer-18
layer-19
layer-20
layer_with_weights-8
layer-21
layer_with_weights-9
layer-22
layer-23
layer_with_weights-10
layer-24
layer-25
layer-26
layer_with_weights-11
layer-27
layer_with_weights-12
layer-28
layer-29
layer_with_weights-13
layer-30
 layer-31
!layer-32
"layer_with_weights-14
"layer-33
#layer_with_weights-15
#layer-34
$layer-35
%layer_with_weights-16
%layer-36
&layer-37
'layer-38
(layer_with_weights-17
(layer-39
)layer_with_weights-18
)layer-40
*layer-41
+layer_with_weights-19
+layer-42
,layer-43
-layer-44
.layer_with_weights-20
.layer-45
/layer_with_weights-21
/layer-46
0layer-47
1layer_with_weights-22
1layer-48
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses
8_default_save_signature
9	optimizer
:
signatures
#;_self_saveable_object_factories"
_tf_keras_network
D
#<_self_saveable_object_factories"
_tf_keras_input_layer
?
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias
#E_self_saveable_object_factories
 F_jit_compiled_convolution_op"
_tf_keras_layer
?
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
#R_self_saveable_object_factories"
_tf_keras_layer
?
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
#Y_self_saveable_object_factories"
_tf_keras_layer
?
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
^__call__
*_&call_and_return_all_conditional_losses
#`_self_saveable_object_factories"
_tf_keras_layer
?
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses
g_random_generator
#h_self_saveable_object_factories"
_tf_keras_layer
?
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses

okernel
pbias
#q_self_saveable_object_factories
 r_jit_compiled_convolution_op"
_tf_keras_layer
?
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses
yaxis
	zgamma
{beta
|moving_mean
}moving_variance
#~_self_saveable_object_factories"
_tf_keras_layer
?
	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_random_generator
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
$?_self_saveable_object_factories"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias
$?_self_saveable_object_factories
!?_jit_compiled_convolution_op"
_tf_keras_layer
?
C0
D1
N2
O3
P4
Q5
o6
p7
z8
{9
|10
}11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45
?46
?47
?48
?49
?50
?51
?52
?53
?54
?55
?56
?57
?58
?59
?60
?61
?62
?63"
trackable_list_wrapper
?
C0
D1
N2
O3
o4
p5
z6
{7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?34
?35
?36
?37
?38
?39
?40
?41
?42
?43
?44
?45"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
8_default_save_signature
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
'__inference_model_2_layer_call_fn_81741
'__inference_model_2_layer_call_fn_83320
'__inference_model_2_layer_call_fn_83453
'__inference_model_2_layer_call_fn_82686?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
B__inference_model_2_layer_call_and_return_conditional_losses_83749
B__inference_model_2_layer_call_and_return_conditional_losses_84101
B__inference_model_2_layer_call_and_return_conditional_losses_82866
B__inference_model_2_layer_call_and_return_conditional_losses_83046?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?B?
 __inference__wrapped_model_80381img"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
	?iter
?beta_1
?beta_2

?decay
?learning_rateCm?Dm?Nm?Om?om?pm?zm?{m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?	?m?Cv?Dv?Nv?Ov?ov?pv?zv?{v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?	?v?"
	optimizer
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_39_layer_call_fn_84110?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_39_layer_call_and_return_conditional_losses_84120?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:(2conv2d_39/kernel
:2conv2d_39/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
<
N0
O1
P2
Q3"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_37_layer_call_fn_84133
6__inference_batch_normalization_37_layer_call_fn_84146?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_84164
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_84182?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:(2batch_normalization_37/gamma
):'2batch_normalization_37/beta
2:0 (2"batch_normalization_37/moving_mean
6:4 (2&batch_normalization_37/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_activation_37_layer_call_fn_84187?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_activation_37_layer_call_and_return_conditional_losses_84192?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
Z	variables
[trainable_variables
\regularization_losses
^__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_max_pooling2d_8_layer_call_fn_84197?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_84202?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_16_layer_call_fn_84207
*__inference_dropout_16_layer_call_fn_84212?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_16_layer_call_and_return_conditional_losses_84217
E__inference_dropout_16_layer_call_and_return_conditional_losses_84229?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_41_layer_call_fn_84238?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_41_layer_call_and_return_conditional_losses_84248?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:( 2conv2d_41/kernel
: 2conv2d_41/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
<
z0
{1
|2
}3"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_39_layer_call_fn_84261
6__inference_batch_normalization_39_layer_call_fn_84274?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_84292
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_84310?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_39/gamma
):' 2batch_normalization_39/beta
2:0  (2"batch_normalization_39/moving_mean
6:4  (2&batch_normalization_39/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_activation_39_layer_call_fn_84315?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_activation_39_layer_call_and_return_conditional_losses_84320?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
/__inference_max_pooling2d_9_layer_call_fn_84325?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_84330?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_17_layer_call_fn_84335
*__inference_dropout_17_layer_call_fn_84340?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_17_layer_call_and_return_conditional_losses_84345
E__inference_dropout_17_layer_call_and_return_conditional_losses_84357?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_43_layer_call_fn_84366?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_43_layer_call_and_return_conditional_losses_84376?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:( @2conv2d_43/kernel
:@2conv2d_43/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_41_layer_call_fn_84389
6__inference_batch_normalization_41_layer_call_fn_84402?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_84420
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_84438?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_41/gamma
):'@2batch_normalization_41/beta
2:0@ (2"batch_normalization_41/moving_mean
6:4@ (2&batch_normalization_41/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_activation_41_layer_call_fn_84443?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_activation_41_layer_call_and_return_conditional_losses_84448?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
0__inference_max_pooling2d_10_layer_call_fn_84453?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_84458?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_18_layer_call_fn_84463
*__inference_dropout_18_layer_call_fn_84468?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_18_layer_call_and_return_conditional_losses_84473
E__inference_dropout_18_layer_call_and_return_conditional_losses_84485?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_45_layer_call_fn_84494?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_45_layer_call_and_return_conditional_losses_84504?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
+:)@?2conv2d_45/kernel
:?2conv2d_45/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_43_layer_call_fn_84517
6__inference_batch_normalization_43_layer_call_fn_84530?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_84548
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_84566?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
+:)?2batch_normalization_43/gamma
*:(?2batch_normalization_43/beta
3:1? (2"batch_normalization_43/moving_mean
7:5? (2&batch_normalization_43/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_activation_43_layer_call_fn_84571?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_activation_43_layer_call_and_return_conditional_losses_84576?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
0__inference_max_pooling2d_11_layer_call_fn_84581?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_84586?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_19_layer_call_fn_84591
*__inference_dropout_19_layer_call_fn_84596?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_19_layer_call_and_return_conditional_losses_84601
E__inference_dropout_19_layer_call_and_return_conditional_losses_84613?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_47_layer_call_fn_84622?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_47_layer_call_and_return_conditional_losses_84632?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
,:*??2conv2d_47/kernel
:?2conv2d_47/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_45_layer_call_fn_84645
6__inference_batch_normalization_45_layer_call_fn_84658?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_84676
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_84694?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
+:)?2batch_normalization_45/gamma
*:(?2batch_normalization_45/beta
3:1? (2"batch_normalization_45/moving_mean
7:5? (2&batch_normalization_45/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_activation_45_layer_call_fn_84699?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_activation_45_layer_call_and_return_conditional_losses_84704?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
2__inference_conv2d_transpose_8_layer_call_fn_84713?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_84746?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
5:3??2conv2d_transpose_8/kernel
&:$?2conv2d_transpose_8/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_concatenate_8_layer_call_fn_84752?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_concatenate_8_layer_call_and_return_conditional_losses_84759?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_20_layer_call_fn_84764
*__inference_dropout_20_layer_call_fn_84769?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_20_layer_call_and_return_conditional_losses_84774
E__inference_dropout_20_layer_call_and_return_conditional_losses_84786?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_49_layer_call_fn_84795?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_49_layer_call_and_return_conditional_losses_84805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
,:*??2conv2d_49/kernel
:?2conv2d_49/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_47_layer_call_fn_84818
6__inference_batch_normalization_47_layer_call_fn_84831?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_84849
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_84867?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
+:)?2batch_normalization_47/gamma
*:(?2batch_normalization_47/beta
3:1? (2"batch_normalization_47/moving_mean
7:5? (2&batch_normalization_47/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_activation_47_layer_call_fn_84872?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_activation_47_layer_call_and_return_conditional_losses_84877?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
2__inference_conv2d_transpose_9_layer_call_fn_84886?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_84919?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
4:2@?2conv2d_transpose_9/kernel
%:#@2conv2d_transpose_9/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_concatenate_9_layer_call_fn_84925?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_concatenate_9_layer_call_and_return_conditional_losses_84932?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_21_layer_call_fn_84937
*__inference_dropout_21_layer_call_fn_84942?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_21_layer_call_and_return_conditional_losses_84947
E__inference_dropout_21_layer_call_and_return_conditional_losses_84959?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_51_layer_call_fn_84968?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_51_layer_call_and_return_conditional_losses_84978?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
+:)?@2conv2d_51/kernel
:@2conv2d_51/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_49_layer_call_fn_84991
6__inference_batch_normalization_49_layer_call_fn_85004?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_85022
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_85040?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_49/gamma
):'@2batch_normalization_49/beta
2:0@ (2"batch_normalization_49/moving_mean
6:4@ (2&batch_normalization_49/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_activation_49_layer_call_fn_85045?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_activation_49_layer_call_and_return_conditional_losses_85050?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_conv2d_transpose_10_layer_call_fn_85059?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_85092?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
4:2 @2conv2d_transpose_10/kernel
&:$ 2conv2d_transpose_10/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
.__inference_concatenate_10_layer_call_fn_85098?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
I__inference_concatenate_10_layer_call_and_return_conditional_losses_85105?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_22_layer_call_fn_85110
*__inference_dropout_22_layer_call_fn_85115?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_22_layer_call_and_return_conditional_losses_85120
E__inference_dropout_22_layer_call_and_return_conditional_losses_85132?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_53_layer_call_fn_85141?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_53_layer_call_and_return_conditional_losses_85151?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:(@ 2conv2d_53/kernel
: 2conv2d_53/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_51_layer_call_fn_85164
6__inference_batch_normalization_51_layer_call_fn_85177?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_85195
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_85213?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:( 2batch_normalization_51/gamma
):' 2batch_normalization_51/beta
2:0  (2"batch_normalization_51/moving_mean
6:4  (2&batch_normalization_51/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_activation_51_layer_call_fn_85218?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_activation_51_layer_call_and_return_conditional_losses_85223?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
3__inference_conv2d_transpose_11_layer_call_fn_85232?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_85265?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
4:2 2conv2d_transpose_11/kernel
&:$2conv2d_transpose_11/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
.__inference_concatenate_11_layer_call_fn_85271?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
I__inference_concatenate_11_layer_call_and_return_conditional_losses_85278?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
*__inference_dropout_23_layer_call_fn_85283
*__inference_dropout_23_layer_call_fn_85288?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
E__inference_dropout_23_layer_call_and_return_conditional_losses_85293
E__inference_dropout_23_layer_call_and_return_conditional_losses_85305?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
"
_generic_user_object
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_55_layer_call_fn_85314?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_55_layer_call_and_return_conditional_losses_85324?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:( 2conv2d_55/kernel
:2conv2d_55/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
6__inference_batch_normalization_53_layer_call_fn_85337
6__inference_batch_normalization_53_layer_call_fn_85350?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_85368
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_85386?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:(2batch_normalization_53/gamma
):'2batch_normalization_53/beta
2:0 (2"batch_normalization_53/moving_mean
6:4 (2&batch_normalization_53/moving_variance
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
-__inference_activation_53_layer_call_fn_85391?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
H__inference_activation_53_layer_call_and_return_conditional_losses_85396?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
)__inference_conv2d_56_layer_call_fn_85405?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
D__inference_conv2d_56_layer_call_and_return_conditional_losses_85416?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:(2conv2d_56/kernel
:2conv2d_56/bias
 "
trackable_dict_wrapper
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
?
P0
Q1
|2
}3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15
?16
?17"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
-44
.45
/46
047
148"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
'__inference_model_2_layer_call_fn_81741img"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
'__inference_model_2_layer_call_fn_83320inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
'__inference_model_2_layer_call_fn_83453inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
'__inference_model_2_layer_call_fn_82686img"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_model_2_layer_call_and_return_conditional_losses_83749inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_model_2_layer_call_and_return_conditional_losses_84101inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_model_2_layer_call_and_return_conditional_losses_82866img"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
B__inference_model_2_layer_call_and_return_conditional_losses_83046img"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
#__inference_signature_wrapper_83187img"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
)__inference_conv2d_39_layer_call_fn_84110inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_39_layer_call_and_return_conditional_losses_84120inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
6__inference_batch_normalization_37_layer_call_fn_84133inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
6__inference_batch_normalization_37_layer_call_fn_84146inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_84164inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_84182inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
-__inference_activation_37_layer_call_fn_84187inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_activation_37_layer_call_and_return_conditional_losses_84192inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
/__inference_max_pooling2d_8_layer_call_fn_84197inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_84202inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
*__inference_dropout_16_layer_call_fn_84207inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_dropout_16_layer_call_fn_84212inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_16_layer_call_and_return_conditional_losses_84217inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_16_layer_call_and_return_conditional_losses_84229inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
)__inference_conv2d_41_layer_call_fn_84238inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_41_layer_call_and_return_conditional_losses_84248inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
6__inference_batch_normalization_39_layer_call_fn_84261inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
6__inference_batch_normalization_39_layer_call_fn_84274inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_84292inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_84310inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
-__inference_activation_39_layer_call_fn_84315inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_activation_39_layer_call_and_return_conditional_losses_84320inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
/__inference_max_pooling2d_9_layer_call_fn_84325inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_84330inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
*__inference_dropout_17_layer_call_fn_84335inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_dropout_17_layer_call_fn_84340inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_17_layer_call_and_return_conditional_losses_84345inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_17_layer_call_and_return_conditional_losses_84357inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
)__inference_conv2d_43_layer_call_fn_84366inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_43_layer_call_and_return_conditional_losses_84376inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
6__inference_batch_normalization_41_layer_call_fn_84389inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
6__inference_batch_normalization_41_layer_call_fn_84402inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_84420inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_84438inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
-__inference_activation_41_layer_call_fn_84443inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_activation_41_layer_call_and_return_conditional_losses_84448inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
0__inference_max_pooling2d_10_layer_call_fn_84453inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_84458inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
*__inference_dropout_18_layer_call_fn_84463inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_dropout_18_layer_call_fn_84468inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_18_layer_call_and_return_conditional_losses_84473inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_18_layer_call_and_return_conditional_losses_84485inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
)__inference_conv2d_45_layer_call_fn_84494inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_45_layer_call_and_return_conditional_losses_84504inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
6__inference_batch_normalization_43_layer_call_fn_84517inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
6__inference_batch_normalization_43_layer_call_fn_84530inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_84548inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_84566inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
-__inference_activation_43_layer_call_fn_84571inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_activation_43_layer_call_and_return_conditional_losses_84576inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
0__inference_max_pooling2d_11_layer_call_fn_84581inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_84586inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
*__inference_dropout_19_layer_call_fn_84591inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_dropout_19_layer_call_fn_84596inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_19_layer_call_and_return_conditional_losses_84601inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_19_layer_call_and_return_conditional_losses_84613inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
)__inference_conv2d_47_layer_call_fn_84622inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_47_layer_call_and_return_conditional_losses_84632inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
6__inference_batch_normalization_45_layer_call_fn_84645inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
6__inference_batch_normalization_45_layer_call_fn_84658inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_84676inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_84694inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
-__inference_activation_45_layer_call_fn_84699inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_activation_45_layer_call_and_return_conditional_losses_84704inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
2__inference_conv2d_transpose_8_layer_call_fn_84713inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_84746inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
-__inference_concatenate_8_layer_call_fn_84752inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_concatenate_8_layer_call_and_return_conditional_losses_84759inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
*__inference_dropout_20_layer_call_fn_84764inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_dropout_20_layer_call_fn_84769inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_20_layer_call_and_return_conditional_losses_84774inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_20_layer_call_and_return_conditional_losses_84786inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
)__inference_conv2d_49_layer_call_fn_84795inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_49_layer_call_and_return_conditional_losses_84805inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
6__inference_batch_normalization_47_layer_call_fn_84818inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
6__inference_batch_normalization_47_layer_call_fn_84831inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_84849inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_84867inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
-__inference_activation_47_layer_call_fn_84872inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_activation_47_layer_call_and_return_conditional_losses_84877inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
2__inference_conv2d_transpose_9_layer_call_fn_84886inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_84919inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
-__inference_concatenate_9_layer_call_fn_84925inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_concatenate_9_layer_call_and_return_conditional_losses_84932inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
*__inference_dropout_21_layer_call_fn_84937inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_dropout_21_layer_call_fn_84942inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_21_layer_call_and_return_conditional_losses_84947inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_21_layer_call_and_return_conditional_losses_84959inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
)__inference_conv2d_51_layer_call_fn_84968inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_51_layer_call_and_return_conditional_losses_84978inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
6__inference_batch_normalization_49_layer_call_fn_84991inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
6__inference_batch_normalization_49_layer_call_fn_85004inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_85022inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_85040inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
-__inference_activation_49_layer_call_fn_85045inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_activation_49_layer_call_and_return_conditional_losses_85050inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
3__inference_conv2d_transpose_10_layer_call_fn_85059inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_85092inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
.__inference_concatenate_10_layer_call_fn_85098inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_concatenate_10_layer_call_and_return_conditional_losses_85105inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
*__inference_dropout_22_layer_call_fn_85110inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_dropout_22_layer_call_fn_85115inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_22_layer_call_and_return_conditional_losses_85120inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_22_layer_call_and_return_conditional_losses_85132inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
)__inference_conv2d_53_layer_call_fn_85141inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_53_layer_call_and_return_conditional_losses_85151inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
6__inference_batch_normalization_51_layer_call_fn_85164inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
6__inference_batch_normalization_51_layer_call_fn_85177inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_85195inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_85213inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
-__inference_activation_51_layer_call_fn_85218inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_activation_51_layer_call_and_return_conditional_losses_85223inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
3__inference_conv2d_transpose_11_layer_call_fn_85232inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_85265inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
.__inference_concatenate_11_layer_call_fn_85271inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
I__inference_concatenate_11_layer_call_and_return_conditional_losses_85278inputs/0inputs/1"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
*__inference_dropout_23_layer_call_fn_85283inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
*__inference_dropout_23_layer_call_fn_85288inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_23_layer_call_and_return_conditional_losses_85293inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
E__inference_dropout_23_layer_call_and_return_conditional_losses_85305inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
)__inference_conv2d_55_layer_call_fn_85314inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_55_layer_call_and_return_conditional_losses_85324inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
6__inference_batch_normalization_53_layer_call_fn_85337inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
6__inference_batch_normalization_53_layer_call_fn_85350inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_85368inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_85386inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
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
?B?
-__inference_activation_53_layer_call_fn_85391inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
H__inference_activation_53_layer_call_and_return_conditional_losses_85396inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
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
?B?
)__inference_conv2d_56_layer_call_fn_85405inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
D__inference_conv2d_56_layer_call_and_return_conditional_losses_85416inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/:-2Adam/conv2d_39/kernel/m
!:2Adam/conv2d_39/bias/m
/:-2#Adam/batch_normalization_37/gamma/m
.:,2"Adam/batch_normalization_37/beta/m
/:- 2Adam/conv2d_41/kernel/m
!: 2Adam/conv2d_41/bias/m
/:- 2#Adam/batch_normalization_39/gamma/m
.:, 2"Adam/batch_normalization_39/beta/m
/:- @2Adam/conv2d_43/kernel/m
!:@2Adam/conv2d_43/bias/m
/:-@2#Adam/batch_normalization_41/gamma/m
.:,@2"Adam/batch_normalization_41/beta/m
0:.@?2Adam/conv2d_45/kernel/m
": ?2Adam/conv2d_45/bias/m
0:.?2#Adam/batch_normalization_43/gamma/m
/:-?2"Adam/batch_normalization_43/beta/m
1:/??2Adam/conv2d_47/kernel/m
": ?2Adam/conv2d_47/bias/m
0:.?2#Adam/batch_normalization_45/gamma/m
/:-?2"Adam/batch_normalization_45/beta/m
::8??2 Adam/conv2d_transpose_8/kernel/m
+:)?2Adam/conv2d_transpose_8/bias/m
1:/??2Adam/conv2d_49/kernel/m
": ?2Adam/conv2d_49/bias/m
0:.?2#Adam/batch_normalization_47/gamma/m
/:-?2"Adam/batch_normalization_47/beta/m
9:7@?2 Adam/conv2d_transpose_9/kernel/m
*:(@2Adam/conv2d_transpose_9/bias/m
0:.?@2Adam/conv2d_51/kernel/m
!:@2Adam/conv2d_51/bias/m
/:-@2#Adam/batch_normalization_49/gamma/m
.:,@2"Adam/batch_normalization_49/beta/m
9:7 @2!Adam/conv2d_transpose_10/kernel/m
+:) 2Adam/conv2d_transpose_10/bias/m
/:-@ 2Adam/conv2d_53/kernel/m
!: 2Adam/conv2d_53/bias/m
/:- 2#Adam/batch_normalization_51/gamma/m
.:, 2"Adam/batch_normalization_51/beta/m
9:7 2!Adam/conv2d_transpose_11/kernel/m
+:)2Adam/conv2d_transpose_11/bias/m
/:- 2Adam/conv2d_55/kernel/m
!:2Adam/conv2d_55/bias/m
/:-2#Adam/batch_normalization_53/gamma/m
.:,2"Adam/batch_normalization_53/beta/m
/:-2Adam/conv2d_56/kernel/m
!:2Adam/conv2d_56/bias/m
/:-2Adam/conv2d_39/kernel/v
!:2Adam/conv2d_39/bias/v
/:-2#Adam/batch_normalization_37/gamma/v
.:,2"Adam/batch_normalization_37/beta/v
/:- 2Adam/conv2d_41/kernel/v
!: 2Adam/conv2d_41/bias/v
/:- 2#Adam/batch_normalization_39/gamma/v
.:, 2"Adam/batch_normalization_39/beta/v
/:- @2Adam/conv2d_43/kernel/v
!:@2Adam/conv2d_43/bias/v
/:-@2#Adam/batch_normalization_41/gamma/v
.:,@2"Adam/batch_normalization_41/beta/v
0:.@?2Adam/conv2d_45/kernel/v
": ?2Adam/conv2d_45/bias/v
0:.?2#Adam/batch_normalization_43/gamma/v
/:-?2"Adam/batch_normalization_43/beta/v
1:/??2Adam/conv2d_47/kernel/v
": ?2Adam/conv2d_47/bias/v
0:.?2#Adam/batch_normalization_45/gamma/v
/:-?2"Adam/batch_normalization_45/beta/v
::8??2 Adam/conv2d_transpose_8/kernel/v
+:)?2Adam/conv2d_transpose_8/bias/v
1:/??2Adam/conv2d_49/kernel/v
": ?2Adam/conv2d_49/bias/v
0:.?2#Adam/batch_normalization_47/gamma/v
/:-?2"Adam/batch_normalization_47/beta/v
9:7@?2 Adam/conv2d_transpose_9/kernel/v
*:(@2Adam/conv2d_transpose_9/bias/v
0:.?@2Adam/conv2d_51/kernel/v
!:@2Adam/conv2d_51/bias/v
/:-@2#Adam/batch_normalization_49/gamma/v
.:,@2"Adam/batch_normalization_49/beta/v
9:7 @2!Adam/conv2d_transpose_10/kernel/v
+:) 2Adam/conv2d_transpose_10/bias/v
/:-@ 2Adam/conv2d_53/kernel/v
!: 2Adam/conv2d_53/bias/v
/:- 2#Adam/batch_normalization_51/gamma/v
.:, 2"Adam/batch_normalization_51/beta/v
9:7 2!Adam/conv2d_transpose_11/kernel/v
+:)2Adam/conv2d_transpose_11/bias/v
/:- 2Adam/conv2d_55/kernel/v
!:2Adam/conv2d_55/bias/v
/:-2#Adam/batch_normalization_53/gamma/v
.:,2"Adam/batch_normalization_53/beta/v
/:-2Adam/conv2d_56/kernel/v
!:2Adam/conv2d_56/bias/v?
 __inference__wrapped_model_80381?tCDNOPQopz{|}????????????????????????????????????????????????????6?3
,?)
'?$
img???????????
? "??<
:
	conv2d_56-?*
	conv2d_56????????????
H__inference_activation_37_layer_call_and_return_conditional_losses_84192l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
-__inference_activation_37_layer_call_fn_84187_9?6
/?,
*?'
inputs???????????
? ""?????????????
H__inference_activation_39_layer_call_and_return_conditional_losses_84320l9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
-__inference_activation_39_layer_call_fn_84315_9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
H__inference_activation_41_layer_call_and_return_conditional_losses_84448h7?4
-?*
(?%
inputs?????????@@@
? "-?*
#? 
0?????????@@@
? ?
-__inference_activation_41_layer_call_fn_84443[7?4
-?*
(?%
inputs?????????@@@
? " ??????????@@@?
H__inference_activation_43_layer_call_and_return_conditional_losses_84576j8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????  ?
? ?
-__inference_activation_43_layer_call_fn_84571]8?5
.?+
)?&
inputs?????????  ?
? "!??????????  ??
H__inference_activation_45_layer_call_and_return_conditional_losses_84704j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_activation_45_layer_call_fn_84699]8?5
.?+
)?&
inputs??????????
? "!????????????
H__inference_activation_47_layer_call_and_return_conditional_losses_84877j8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????  ?
? ?
-__inference_activation_47_layer_call_fn_84872]8?5
.?+
)?&
inputs?????????  ?
? "!??????????  ??
H__inference_activation_49_layer_call_and_return_conditional_losses_85050h7?4
-?*
(?%
inputs?????????@@@
? "-?*
#? 
0?????????@@@
? ?
-__inference_activation_49_layer_call_fn_85045[7?4
-?*
(?%
inputs?????????@@@
? " ??????????@@@?
H__inference_activation_51_layer_call_and_return_conditional_losses_85223l9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0??????????? 
? ?
-__inference_activation_51_layer_call_fn_85218_9?6
/?,
*?'
inputs??????????? 
? ""???????????? ?
H__inference_activation_53_layer_call_and_return_conditional_losses_85396l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
-__inference_activation_53_layer_call_fn_85391_9?6
/?,
*?'
inputs???????????
? ""?????????????
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_84164?NOPQM?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_37_layer_call_and_return_conditional_losses_84182?NOPQM?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
6__inference_batch_normalization_37_layer_call_fn_84133?NOPQM?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_37_layer_call_fn_84146?NOPQM?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_84292?z{|}M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_39_layer_call_and_return_conditional_losses_84310?z{|}M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
6__inference_batch_normalization_39_layer_call_fn_84261?z{|}M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_39_layer_call_fn_84274?z{|}M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_84420?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_41_layer_call_and_return_conditional_losses_84438?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
6__inference_batch_normalization_41_layer_call_fn_84389?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
6__inference_batch_normalization_41_layer_call_fn_84402?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_84548?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_43_layer_call_and_return_conditional_losses_84566?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_43_layer_call_fn_84517?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_43_layer_call_fn_84530?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_84676?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_45_layer_call_and_return_conditional_losses_84694?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_45_layer_call_fn_84645?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_45_layer_call_fn_84658?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_84849?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
Q__inference_batch_normalization_47_layer_call_and_return_conditional_losses_84867?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
6__inference_batch_normalization_47_layer_call_fn_84818?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
6__inference_batch_normalization_47_layer_call_fn_84831?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_85022?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
Q__inference_batch_normalization_49_layer_call_and_return_conditional_losses_85040?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
6__inference_batch_normalization_49_layer_call_fn_84991?????M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
6__inference_batch_normalization_49_layer_call_fn_85004?????M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_85195?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
Q__inference_batch_normalization_51_layer_call_and_return_conditional_losses_85213?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
6__inference_batch_normalization_51_layer_call_fn_85164?????M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
6__inference_batch_normalization_51_layer_call_fn_85177?????M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_85368?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
Q__inference_batch_normalization_53_layer_call_and_return_conditional_losses_85386?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
6__inference_batch_normalization_53_layer_call_fn_85337?????M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
6__inference_batch_normalization_53_layer_call_fn_85350?????M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
I__inference_concatenate_10_layer_call_and_return_conditional_losses_85105?n?k
d?a
_?\
,?)
inputs/0??????????? 
,?)
inputs/1??????????? 
? "/?,
%?"
0???????????@
? ?
.__inference_concatenate_10_layer_call_fn_85098?n?k
d?a
_?\
,?)
inputs/0??????????? 
,?)
inputs/1??????????? 
? ""????????????@?
I__inference_concatenate_11_layer_call_and_return_conditional_losses_85278?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? "/?,
%?"
0??????????? 
? ?
.__inference_concatenate_11_layer_call_fn_85271?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? ""???????????? ?
H__inference_concatenate_8_layer_call_and_return_conditional_losses_84759?l?i
b?_
]?Z
+?(
inputs/0?????????  ?
+?(
inputs/1?????????  ?
? ".?+
$?!
0?????????  ?
? ?
-__inference_concatenate_8_layer_call_fn_84752?l?i
b?_
]?Z
+?(
inputs/0?????????  ?
+?(
inputs/1?????????  ?
? "!??????????  ??
H__inference_concatenate_9_layer_call_and_return_conditional_losses_84932?j?g
`?]
[?X
*?'
inputs/0?????????@@@
*?'
inputs/1?????????@@@
? ".?+
$?!
0?????????@@?
? ?
-__inference_concatenate_9_layer_call_fn_84925?j?g
`?]
[?X
*?'
inputs/0?????????@@@
*?'
inputs/1?????????@@@
? "!??????????@@??
D__inference_conv2d_39_layer_call_and_return_conditional_losses_84120pCD9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
)__inference_conv2d_39_layer_call_fn_84110cCD9?6
/?,
*?'
inputs???????????
? ""?????????????
D__inference_conv2d_41_layer_call_and_return_conditional_losses_84248pop9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0??????????? 
? ?
)__inference_conv2d_41_layer_call_fn_84238cop9?6
/?,
*?'
inputs???????????
? ""???????????? ?
D__inference_conv2d_43_layer_call_and_return_conditional_losses_84376n??7?4
-?*
(?%
inputs?????????@@ 
? "-?*
#? 
0?????????@@@
? ?
)__inference_conv2d_43_layer_call_fn_84366a??7?4
-?*
(?%
inputs?????????@@ 
? " ??????????@@@?
D__inference_conv2d_45_layer_call_and_return_conditional_losses_84504o??7?4
-?*
(?%
inputs?????????  @
? ".?+
$?!
0?????????  ?
? ?
)__inference_conv2d_45_layer_call_fn_84494b??7?4
-?*
(?%
inputs?????????  @
? "!??????????  ??
D__inference_conv2d_47_layer_call_and_return_conditional_losses_84632p??8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_47_layer_call_fn_84622c??8?5
.?+
)?&
inputs??????????
? "!????????????
D__inference_conv2d_49_layer_call_and_return_conditional_losses_84805p??8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????  ?
? ?
)__inference_conv2d_49_layer_call_fn_84795c??8?5
.?+
)?&
inputs?????????  ?
? "!??????????  ??
D__inference_conv2d_51_layer_call_and_return_conditional_losses_84978o??8?5
.?+
)?&
inputs?????????@@?
? "-?*
#? 
0?????????@@@
? ?
)__inference_conv2d_51_layer_call_fn_84968b??8?5
.?+
)?&
inputs?????????@@?
? " ??????????@@@?
D__inference_conv2d_53_layer_call_and_return_conditional_losses_85151r??9?6
/?,
*?'
inputs???????????@
? "/?,
%?"
0??????????? 
? ?
)__inference_conv2d_53_layer_call_fn_85141e??9?6
/?,
*?'
inputs???????????@
? ""???????????? ?
D__inference_conv2d_55_layer_call_and_return_conditional_losses_85324r??9?6
/?,
*?'
inputs??????????? 
? "/?,
%?"
0???????????
? ?
)__inference_conv2d_55_layer_call_fn_85314e??9?6
/?,
*?'
inputs??????????? 
? ""?????????????
D__inference_conv2d_56_layer_call_and_return_conditional_losses_85416r??9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
)__inference_conv2d_56_layer_call_fn_85405e??9?6
/?,
*?'
inputs???????????
? ""?????????????
N__inference_conv2d_transpose_10_layer_call_and_return_conditional_losses_85092???I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
3__inference_conv2d_transpose_10_layer_call_fn_85059???I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
N__inference_conv2d_transpose_11_layer_call_and_return_conditional_losses_85265???I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
3__inference_conv2d_transpose_11_layer_call_fn_85232???I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
M__inference_conv2d_transpose_8_layer_call_and_return_conditional_losses_84746???J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
2__inference_conv2d_transpose_8_layer_call_fn_84713???J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
M__inference_conv2d_transpose_9_layer_call_and_return_conditional_losses_84919???J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????@
? ?
2__inference_conv2d_transpose_9_layer_call_fn_84886???J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+???????????????????????????@?
E__inference_dropout_16_layer_call_and_return_conditional_losses_84217p=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
E__inference_dropout_16_layer_call_and_return_conditional_losses_84229p=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
*__inference_dropout_16_layer_call_fn_84207c=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
*__inference_dropout_16_layer_call_fn_84212c=?:
3?0
*?'
inputs???????????
p
? ""?????????????
E__inference_dropout_17_layer_call_and_return_conditional_losses_84345l;?8
1?.
(?%
inputs?????????@@ 
p 
? "-?*
#? 
0?????????@@ 
? ?
E__inference_dropout_17_layer_call_and_return_conditional_losses_84357l;?8
1?.
(?%
inputs?????????@@ 
p
? "-?*
#? 
0?????????@@ 
? ?
*__inference_dropout_17_layer_call_fn_84335_;?8
1?.
(?%
inputs?????????@@ 
p 
? " ??????????@@ ?
*__inference_dropout_17_layer_call_fn_84340_;?8
1?.
(?%
inputs?????????@@ 
p
? " ??????????@@ ?
E__inference_dropout_18_layer_call_and_return_conditional_losses_84473l;?8
1?.
(?%
inputs?????????  @
p 
? "-?*
#? 
0?????????  @
? ?
E__inference_dropout_18_layer_call_and_return_conditional_losses_84485l;?8
1?.
(?%
inputs?????????  @
p
? "-?*
#? 
0?????????  @
? ?
*__inference_dropout_18_layer_call_fn_84463_;?8
1?.
(?%
inputs?????????  @
p 
? " ??????????  @?
*__inference_dropout_18_layer_call_fn_84468_;?8
1?.
(?%
inputs?????????  @
p
? " ??????????  @?
E__inference_dropout_19_layer_call_and_return_conditional_losses_84601n<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
E__inference_dropout_19_layer_call_and_return_conditional_losses_84613n<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
*__inference_dropout_19_layer_call_fn_84591a<?9
2?/
)?&
inputs??????????
p 
? "!????????????
*__inference_dropout_19_layer_call_fn_84596a<?9
2?/
)?&
inputs??????????
p
? "!????????????
E__inference_dropout_20_layer_call_and_return_conditional_losses_84774n<?9
2?/
)?&
inputs?????????  ?
p 
? ".?+
$?!
0?????????  ?
? ?
E__inference_dropout_20_layer_call_and_return_conditional_losses_84786n<?9
2?/
)?&
inputs?????????  ?
p
? ".?+
$?!
0?????????  ?
? ?
*__inference_dropout_20_layer_call_fn_84764a<?9
2?/
)?&
inputs?????????  ?
p 
? "!??????????  ??
*__inference_dropout_20_layer_call_fn_84769a<?9
2?/
)?&
inputs?????????  ?
p
? "!??????????  ??
E__inference_dropout_21_layer_call_and_return_conditional_losses_84947n<?9
2?/
)?&
inputs?????????@@?
p 
? ".?+
$?!
0?????????@@?
? ?
E__inference_dropout_21_layer_call_and_return_conditional_losses_84959n<?9
2?/
)?&
inputs?????????@@?
p
? ".?+
$?!
0?????????@@?
? ?
*__inference_dropout_21_layer_call_fn_84937a<?9
2?/
)?&
inputs?????????@@?
p 
? "!??????????@@??
*__inference_dropout_21_layer_call_fn_84942a<?9
2?/
)?&
inputs?????????@@?
p
? "!??????????@@??
E__inference_dropout_22_layer_call_and_return_conditional_losses_85120p=?:
3?0
*?'
inputs???????????@
p 
? "/?,
%?"
0???????????@
? ?
E__inference_dropout_22_layer_call_and_return_conditional_losses_85132p=?:
3?0
*?'
inputs???????????@
p
? "/?,
%?"
0???????????@
? ?
*__inference_dropout_22_layer_call_fn_85110c=?:
3?0
*?'
inputs???????????@
p 
? ""????????????@?
*__inference_dropout_22_layer_call_fn_85115c=?:
3?0
*?'
inputs???????????@
p
? ""????????????@?
E__inference_dropout_23_layer_call_and_return_conditional_losses_85293p=?:
3?0
*?'
inputs??????????? 
p 
? "/?,
%?"
0??????????? 
? ?
E__inference_dropout_23_layer_call_and_return_conditional_losses_85305p=?:
3?0
*?'
inputs??????????? 
p
? "/?,
%?"
0??????????? 
? ?
*__inference_dropout_23_layer_call_fn_85283c=?:
3?0
*?'
inputs??????????? 
p 
? ""???????????? ?
*__inference_dropout_23_layer_call_fn_85288c=?:
3?0
*?'
inputs??????????? 
p
? ""???????????? ?
K__inference_max_pooling2d_10_layer_call_and_return_conditional_losses_84458?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_10_layer_call_fn_84453?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
K__inference_max_pooling2d_11_layer_call_and_return_conditional_losses_84586?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_max_pooling2d_11_layer_call_fn_84581?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_84202?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_8_layer_call_fn_84197?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_84330?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_9_layer_call_fn_84325?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
B__inference_model_2_layer_call_and_return_conditional_losses_82866?tCDNOPQopz{|}????????????????????????????????????????????????????>?;
4?1
'?$
img???????????
p 

 
? "/?,
%?"
0???????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_83046?tCDNOPQopz{|}????????????????????????????????????????????????????>?;
4?1
'?$
img???????????
p

 
? "/?,
%?"
0???????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_83749?tCDNOPQopz{|}????????????????????????????????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
B__inference_model_2_layer_call_and_return_conditional_losses_84101?tCDNOPQopz{|}????????????????????????????????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
'__inference_model_2_layer_call_fn_81741?tCDNOPQopz{|}????????????????????????????????????????????????????>?;
4?1
'?$
img???????????
p 

 
? ""?????????????
'__inference_model_2_layer_call_fn_82686?tCDNOPQopz{|}????????????????????????????????????????????????????>?;
4?1
'?$
img???????????
p

 
? ""?????????????
'__inference_model_2_layer_call_fn_83320?tCDNOPQopz{|}????????????????????????????????????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
'__inference_model_2_layer_call_fn_83453?tCDNOPQopz{|}????????????????????????????????????????????????????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
#__inference_signature_wrapper_83187?tCDNOPQopz{|}????????????????????????????????????????????????????=?:
? 
3?0
.
img'?$
img???????????"??<
:
	conv2d_56-?*
	conv2d_56???????????