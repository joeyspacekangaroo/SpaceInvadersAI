??
??
?
ArgMax

input"T
	dimension"Tidx
output"output_type"!
Ttype:
2	
"
Tidxtype0:
2	"
output_typetype0	:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
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
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
executor_typestring ?
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
Ttype"
Indextype:
2	"

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
 ?"serve*	2.5.0-rc32v2.5.0-rc2-14-gfcdf65934708??
d
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0	
?
)QNetwork/EncodingNetwork/conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)QNetwork/EncodingNetwork/conv2d_24/kernel
?
=QNetwork/EncodingNetwork/conv2d_24/kernel/Read/ReadVariableOpReadVariableOp)QNetwork/EncodingNetwork/conv2d_24/kernel*&
_output_shapes
: *
dtype0
?
'QNetwork/EncodingNetwork/conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *8
shared_name)'QNetwork/EncodingNetwork/conv2d_24/bias
?
;QNetwork/EncodingNetwork/conv2d_24/bias/Read/ReadVariableOpReadVariableOp'QNetwork/EncodingNetwork/conv2d_24/bias*
_output_shapes
: *
dtype0
?
)QNetwork/EncodingNetwork/conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*:
shared_name+)QNetwork/EncodingNetwork/conv2d_25/kernel
?
=QNetwork/EncodingNetwork/conv2d_25/kernel/Read/ReadVariableOpReadVariableOp)QNetwork/EncodingNetwork/conv2d_25/kernel*&
_output_shapes
: @*
dtype0
?
'QNetwork/EncodingNetwork/conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'QNetwork/EncodingNetwork/conv2d_25/bias
?
;QNetwork/EncodingNetwork/conv2d_25/bias/Read/ReadVariableOpReadVariableOp'QNetwork/EncodingNetwork/conv2d_25/bias*
_output_shapes
:@*
dtype0
?
)QNetwork/EncodingNetwork/conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*:
shared_name+)QNetwork/EncodingNetwork/conv2d_26/kernel
?
=QNetwork/EncodingNetwork/conv2d_26/kernel/Read/ReadVariableOpReadVariableOp)QNetwork/EncodingNetwork/conv2d_26/kernel*&
_output_shapes
:@@*
dtype0
?
'QNetwork/EncodingNetwork/conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*8
shared_name)'QNetwork/EncodingNetwork/conv2d_26/bias
?
;QNetwork/EncodingNetwork/conv2d_26/bias/Read/ReadVariableOpReadVariableOp'QNetwork/EncodingNetwork/conv2d_26/bias*
_output_shapes
:@*
dtype0
?
(QNetwork/EncodingNetwork/dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*9
shared_name*(QNetwork/EncodingNetwork/dense_16/kernel
?
<QNetwork/EncodingNetwork/dense_16/kernel/Read/ReadVariableOpReadVariableOp(QNetwork/EncodingNetwork/dense_16/kernel* 
_output_shapes
:
??*
dtype0
?
&QNetwork/EncodingNetwork/dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&QNetwork/EncodingNetwork/dense_16/bias
?
:QNetwork/EncodingNetwork/dense_16/bias/Read/ReadVariableOpReadVariableOp&QNetwork/EncodingNetwork/dense_16/bias*
_output_shapes	
:?*
dtype0
?
QNetwork/dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*)
shared_nameQNetwork/dense_17/kernel
?
,QNetwork/dense_17/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense_17/kernel*
_output_shapes
:	?*
dtype0
?
QNetwork/dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameQNetwork/dense_17/bias
}
*QNetwork/dense_17/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense_17/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?#
value?#B?# B?#
T

train_step
metadata
model_variables
_all_assets

signatures
CA
VARIABLE_VALUEVariable%train_step/.ATTRIBUTES/VARIABLE_VALUE
 
F
0
1
2
	3

4
5
6
7
8
9

0
 
ki
VARIABLE_VALUE)QNetwork/EncodingNetwork/conv2d_24/kernel,model_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE'QNetwork/EncodingNetwork/conv2d_24/bias,model_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE)QNetwork/EncodingNetwork/conv2d_25/kernel,model_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE'QNetwork/EncodingNetwork/conv2d_25/bias,model_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE)QNetwork/EncodingNetwork/conv2d_26/kernel,model_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE'QNetwork/EncodingNetwork/conv2d_26/bias,model_variables/5/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE(QNetwork/EncodingNetwork/dense_16/kernel,model_variables/6/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE&QNetwork/EncodingNetwork/dense_16/bias,model_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEQNetwork/dense_17/kernel,model_variables/8/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEQNetwork/dense_17/bias,model_variables/9/.ATTRIBUTES/VARIABLE_VALUE

ref
1


_q_network
t
_encoder
_q_value_layer
regularization_losses
trainable_variables
	variables
	keras_api
?
_flat_preprocessing_layers
_postprocessing_layers
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
 
F
0
1
2
	3

4
5
6
7
8
9
F
0
1
2
	3

4
5
6
7
8
9
?
#metrics
$layer_metrics

%layers
regularization_losses
trainable_variables
	variables
&non_trainable_variables
'layer_regularization_losses

(0
#
)0
*1
+2
,3
-4
 
8
0
1
2
	3

4
5
6
7
8
0
1
2
	3

4
5
6
7
?
.metrics
/layer_metrics

0layers
regularization_losses
trainable_variables
	variables
1non_trainable_variables
2layer_regularization_losses
 

0
1

0
1
?
3metrics
4layer_metrics

5layers
regularization_losses
 trainable_variables
!	variables
6non_trainable_variables
7layer_regularization_losses
 
 

0
1
 
 
R
8regularization_losses
9trainable_variables
:	variables
;	keras_api
h

kernel
bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
h

kernel
	bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
h


kernel
bias
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
R
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
h

kernel
bias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
 
 
*
(0
)1
*2
+3
,4
-5
 
 
 
 
 
 
 
 
 
 
?
Pmetrics
Qlayer_metrics

Rlayers
8regularization_losses
9trainable_variables
:	variables
Snon_trainable_variables
Tlayer_regularization_losses
 

0
1

0
1
?
Umetrics
Vlayer_metrics

Wlayers
<regularization_losses
=trainable_variables
>	variables
Xnon_trainable_variables
Ylayer_regularization_losses
 

0
	1

0
	1
?
Zmetrics
[layer_metrics

\layers
@regularization_losses
Atrainable_variables
B	variables
]non_trainable_variables
^layer_regularization_losses
 


0
1


0
1
?
_metrics
`layer_metrics

alayers
Dregularization_losses
Etrainable_variables
F	variables
bnon_trainable_variables
clayer_regularization_losses
 
 
 
?
dmetrics
elayer_metrics

flayers
Hregularization_losses
Itrainable_variables
J	variables
gnon_trainable_variables
hlayer_regularization_losses
 

0
1

0
1
?
imetrics
jlayer_metrics

klayers
Lregularization_losses
Mtrainable_variables
N	variables
lnon_trainable_variables
mlayer_regularization_losses
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
action_0/observationPlaceholder*/
_output_shapes
:?????????TT*
dtype0*$
shape:?????????TT
j
action_0/rewardPlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
m
action_0/step_typePlaceholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type)QNetwork/EncodingNetwork/conv2d_24/kernel'QNetwork/EncodingNetwork/conv2d_24/bias)QNetwork/EncodingNetwork/conv2d_25/kernel'QNetwork/EncodingNetwork/conv2d_25/bias)QNetwork/EncodingNetwork/conv2d_26/kernel'QNetwork/EncodingNetwork/conv2d_26/bias(QNetwork/EncodingNetwork/dense_16/kernel&QNetwork/EncodingNetwork/dense_16/biasQNetwork/dense_17/kernelQNetwork/dense_17/bias*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_20466291
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
PartitionedCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_20466296
?
PartitionedCall_1PartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_20466308
?
StatefulPartitionedCall_1StatefulPartitionedCallVariable*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_20466304
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOp=QNetwork/EncodingNetwork/conv2d_24/kernel/Read/ReadVariableOp;QNetwork/EncodingNetwork/conv2d_24/bias/Read/ReadVariableOp=QNetwork/EncodingNetwork/conv2d_25/kernel/Read/ReadVariableOp;QNetwork/EncodingNetwork/conv2d_25/bias/Read/ReadVariableOp=QNetwork/EncodingNetwork/conv2d_26/kernel/Read/ReadVariableOp;QNetwork/EncodingNetwork/conv2d_26/bias/Read/ReadVariableOp<QNetwork/EncodingNetwork/dense_16/kernel/Read/ReadVariableOp:QNetwork/EncodingNetwork/dense_16/bias/Read/ReadVariableOp,QNetwork/dense_17/kernel/Read/ReadVariableOp*QNetwork/dense_17/bias/Read/ReadVariableOpConst*
Tin
2	*
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
!__inference__traced_save_20466369
?
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable)QNetwork/EncodingNetwork/conv2d_24/kernel'QNetwork/EncodingNetwork/conv2d_24/bias)QNetwork/EncodingNetwork/conv2d_25/kernel'QNetwork/EncodingNetwork/conv2d_25/bias)QNetwork/EncodingNetwork/conv2d_26/kernel'QNetwork/EncodingNetwork/conv2d_26/bias(QNetwork/EncodingNetwork/dense_16/kernel&QNetwork/EncodingNetwork/dense_16/biasQNetwork/dense_17/kernelQNetwork/dense_17/bias*
Tin
2*
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
GPU2*0J 8? *-
f(R&
$__inference__traced_restore_20466412??
?
.
,__inference_function_with_signature_20465894?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference_<lambda>_204656072
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
^

__inference_<lambda>_20465607*(
_construction_contextkEagerRuntime*
_input_shapes 
?\
?
0__inference_polymorphic_distribution_fn_20466095
	step_type

reward
discount
observation[
Aqnetwork_encodingnetwork_conv2d_24_conv2d_readvariableop_resource: P
Bqnetwork_encodingnetwork_conv2d_24_biasadd_readvariableop_resource: [
Aqnetwork_encodingnetwork_conv2d_25_conv2d_readvariableop_resource: @P
Bqnetwork_encodingnetwork_conv2d_25_biasadd_readvariableop_resource:@[
Aqnetwork_encodingnetwork_conv2d_26_conv2d_readvariableop_resource:@@P
Bqnetwork_encodingnetwork_conv2d_26_biasadd_readvariableop_resource:@T
@qnetwork_encodingnetwork_dense_16_matmul_readvariableop_resource:
??P
Aqnetwork_encodingnetwork_dense_16_biasadd_readvariableop_resource:	?C
0qnetwork_dense_17_matmul_readvariableop_resource:	??
1qnetwork_dense_17_biasadd_readvariableop_resource:
identity	??9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp?9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp?9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp?8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp?(QNetwork/dense_17/BiasAdd/ReadVariableOp?'QNetwork/dense_17/MatMul/ReadVariableOp?
&QNetwork/EncodingNetwork/lambda_4/CastCastobservation*

DstT0*

SrcT0*/
_output_shapes
:?????????TT2(
&QNetwork/EncodingNetwork/lambda_4/Cast?
+QNetwork/EncodingNetwork/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2-
+QNetwork/EncodingNetwork/lambda_4/truediv/y?
)QNetwork/EncodingNetwork/lambda_4/truedivRealDiv*QNetwork/EncodingNetwork/lambda_4/Cast:y:04QNetwork/EncodingNetwork/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:?????????TT2+
)QNetwork/EncodingNetwork/lambda_4/truediv?
8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02:
8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_24/Conv2DConv2D-QNetwork/EncodingNetwork/lambda_4/truediv:z:0@QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2+
)QNetwork/EncodingNetwork/conv2d_24/Conv2D?
9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOpReadVariableOpBqnetwork_encodingnetwork_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp?
*QNetwork/EncodingNetwork/conv2d_24/BiasAddBiasAdd2QNetwork/EncodingNetwork/conv2d_24/Conv2D:output:0AQNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2,
*QNetwork/EncodingNetwork/conv2d_24/BiasAdd?
'QNetwork/EncodingNetwork/conv2d_24/ReluRelu3QNetwork/EncodingNetwork/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2)
'QNetwork/EncodingNetwork/conv2d_24/Relu?
8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02:
8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_25/Conv2DConv2D5QNetwork/EncodingNetwork/conv2d_24/Relu:activations:0@QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@*
paddingVALID*
strides
2+
)QNetwork/EncodingNetwork/conv2d_25/Conv2D?
9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOpReadVariableOpBqnetwork_encodingnetwork_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp?
*QNetwork/EncodingNetwork/conv2d_25/BiasAddBiasAdd2QNetwork/EncodingNetwork/conv2d_25/Conv2D:output:0AQNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@2,
*QNetwork/EncodingNetwork/conv2d_25/BiasAdd?
'QNetwork/EncodingNetwork/conv2d_25/ReluRelu3QNetwork/EncodingNetwork/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		@2)
'QNetwork/EncodingNetwork/conv2d_25/Relu?
8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02:
8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_26/Conv2DConv2D5QNetwork/EncodingNetwork/conv2d_25/Relu:activations:0@QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2+
)QNetwork/EncodingNetwork/conv2d_26/Conv2D?
9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOpReadVariableOpBqnetwork_encodingnetwork_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp?
*QNetwork/EncodingNetwork/conv2d_26/BiasAddBiasAdd2QNetwork/EncodingNetwork/conv2d_26/Conv2D:output:0AQNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2,
*QNetwork/EncodingNetwork/conv2d_26/BiasAdd?
'QNetwork/EncodingNetwork/conv2d_26/ReluRelu3QNetwork/EncodingNetwork/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2)
'QNetwork/EncodingNetwork/conv2d_26/Relu?
(QNetwork/EncodingNetwork/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2*
(QNetwork/EncodingNetwork/flatten_8/Const?
*QNetwork/EncodingNetwork/flatten_8/ReshapeReshape5QNetwork/EncodingNetwork/conv2d_26/Relu:activations:01QNetwork/EncodingNetwork/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2,
*QNetwork/EncodingNetwork/flatten_8/Reshape?
7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_16/MatMulMatMul3QNetwork/EncodingNetwork/flatten_8/Reshape:output:0?QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_16/MatMul?
8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_16/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_16/MatMul:product:0@QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_16/BiasAdd?
&QNetwork/EncodingNetwork/dense_16/ReluRelu2QNetwork/EncodingNetwork/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_16/Relu?
'QNetwork/dense_17/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'QNetwork/dense_17/MatMul/ReadVariableOp?
QNetwork/dense_17/MatMulMatMul4QNetwork/EncodingNetwork/dense_16/Relu:activations:0/QNetwork/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_17/MatMul?
(QNetwork/dense_17/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_17/BiasAdd/ReadVariableOp?
QNetwork/dense_17/BiasAddBiasAdd"QNetwork/dense_17/MatMul:product:00QNetwork/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_17/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_17/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMaxj
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic/rtoln
Deterministic_1/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic_1/atoln
Deterministic_1/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic_1/rtol?
IdentityIdentity"Categorical_1/mode/ArgMax:output:0:^QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp:^QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp:^QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp9^QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp)^QNetwork/dense_17/BiasAdd/ReadVariableOp(^QNetwork/dense_17/MatMul/ReadVariableOp*
T0	*#
_output_shapes
:?????????2

Identityn
Deterministic_2/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic_2/atoln
Deterministic_2/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic_2/rtol"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:?????????:?????????:?????????:?????????TT: : : : : : : : : : 2v
9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp2v
9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp2v
9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp2T
(QNetwork/dense_17/BiasAdd/ReadVariableOp(QNetwork/dense_17/BiasAdd/ReadVariableOp2R
'QNetwork/dense_17/MatMul/ReadVariableOp'QNetwork/dense_17/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:\X
/
_output_shapes
:?????????TT
%
_user_specified_nameobservation
?
d
__inference_<lambda>_20465604!
readvariableop_resource:	 
identity	??ReadVariableOpp
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0	2
ReadVariableOpj
IdentityIdentityReadVariableOp:value:0^ReadVariableOp*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2 
ReadVariableOpReadVariableOp
?
8
&__inference_signature_wrapper_20466296

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_function_with_signature_204658712
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?v
?
*__inference_polymorphic_action_fn_20466041
time_step_step_type
time_step_reward
time_step_discount
time_step_observation[
Aqnetwork_encodingnetwork_conv2d_24_conv2d_readvariableop_resource: P
Bqnetwork_encodingnetwork_conv2d_24_biasadd_readvariableop_resource: [
Aqnetwork_encodingnetwork_conv2d_25_conv2d_readvariableop_resource: @P
Bqnetwork_encodingnetwork_conv2d_25_biasadd_readvariableop_resource:@[
Aqnetwork_encodingnetwork_conv2d_26_conv2d_readvariableop_resource:@@P
Bqnetwork_encodingnetwork_conv2d_26_biasadd_readvariableop_resource:@T
@qnetwork_encodingnetwork_dense_16_matmul_readvariableop_resource:
??P
Aqnetwork_encodingnetwork_dense_16_biasadd_readvariableop_resource:	?C
0qnetwork_dense_17_matmul_readvariableop_resource:	??
1qnetwork_dense_17_biasadd_readvariableop_resource:
identity	??9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp?9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp?9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp?8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp?(QNetwork/dense_17/BiasAdd/ReadVariableOp?'QNetwork/dense_17/MatMul/ReadVariableOp?
&QNetwork/EncodingNetwork/lambda_4/CastCasttime_step_observation*

DstT0*

SrcT0*/
_output_shapes
:?????????TT2(
&QNetwork/EncodingNetwork/lambda_4/Cast?
+QNetwork/EncodingNetwork/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2-
+QNetwork/EncodingNetwork/lambda_4/truediv/y?
)QNetwork/EncodingNetwork/lambda_4/truedivRealDiv*QNetwork/EncodingNetwork/lambda_4/Cast:y:04QNetwork/EncodingNetwork/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:?????????TT2+
)QNetwork/EncodingNetwork/lambda_4/truediv?
8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02:
8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_24/Conv2DConv2D-QNetwork/EncodingNetwork/lambda_4/truediv:z:0@QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2+
)QNetwork/EncodingNetwork/conv2d_24/Conv2D?
9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOpReadVariableOpBqnetwork_encodingnetwork_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp?
*QNetwork/EncodingNetwork/conv2d_24/BiasAddBiasAdd2QNetwork/EncodingNetwork/conv2d_24/Conv2D:output:0AQNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2,
*QNetwork/EncodingNetwork/conv2d_24/BiasAdd?
'QNetwork/EncodingNetwork/conv2d_24/ReluRelu3QNetwork/EncodingNetwork/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2)
'QNetwork/EncodingNetwork/conv2d_24/Relu?
8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02:
8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_25/Conv2DConv2D5QNetwork/EncodingNetwork/conv2d_24/Relu:activations:0@QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@*
paddingVALID*
strides
2+
)QNetwork/EncodingNetwork/conv2d_25/Conv2D?
9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOpReadVariableOpBqnetwork_encodingnetwork_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp?
*QNetwork/EncodingNetwork/conv2d_25/BiasAddBiasAdd2QNetwork/EncodingNetwork/conv2d_25/Conv2D:output:0AQNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@2,
*QNetwork/EncodingNetwork/conv2d_25/BiasAdd?
'QNetwork/EncodingNetwork/conv2d_25/ReluRelu3QNetwork/EncodingNetwork/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		@2)
'QNetwork/EncodingNetwork/conv2d_25/Relu?
8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02:
8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_26/Conv2DConv2D5QNetwork/EncodingNetwork/conv2d_25/Relu:activations:0@QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2+
)QNetwork/EncodingNetwork/conv2d_26/Conv2D?
9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOpReadVariableOpBqnetwork_encodingnetwork_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp?
*QNetwork/EncodingNetwork/conv2d_26/BiasAddBiasAdd2QNetwork/EncodingNetwork/conv2d_26/Conv2D:output:0AQNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2,
*QNetwork/EncodingNetwork/conv2d_26/BiasAdd?
'QNetwork/EncodingNetwork/conv2d_26/ReluRelu3QNetwork/EncodingNetwork/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2)
'QNetwork/EncodingNetwork/conv2d_26/Relu?
(QNetwork/EncodingNetwork/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2*
(QNetwork/EncodingNetwork/flatten_8/Const?
*QNetwork/EncodingNetwork/flatten_8/ReshapeReshape5QNetwork/EncodingNetwork/conv2d_26/Relu:activations:01QNetwork/EncodingNetwork/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2,
*QNetwork/EncodingNetwork/flatten_8/Reshape?
7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_16/MatMulMatMul3QNetwork/EncodingNetwork/flatten_8/Reshape:output:0?QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_16/MatMul?
8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_16/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_16/MatMul:product:0@QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_16/BiasAdd?
&QNetwork/EncodingNetwork/dense_16/ReluRelu2QNetwork/EncodingNetwork/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_16/Relu?
'QNetwork/dense_17/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'QNetwork/dense_17/MatMul/ReadVariableOp?
QNetwork/dense_17/MatMulMatMul4QNetwork/EncodingNetwork/dense_16/Relu:activations:0/QNetwork/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_17/MatMul?
(QNetwork/dense_17/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_17/BiasAdd/ReadVariableOp?
QNetwork/dense_17/BiasAddBiasAdd"QNetwork/dense_17/MatMul:product:00QNetwork/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_17/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_17/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMaxj
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShape"Categorical_1/mode/ArgMax:output:0*
T0	*
_output_shapes
:2
Deterministic_1/sample/Shape?
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastTo"Categorical_1/mode/ArgMax:output:0&Deterministic_1/sample/concat:output:0*
T0	*'
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0	*#
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:?????????2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:0:^QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp:^QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp:^QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp9^QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp)^QNetwork/dense_17/BiasAdd/ReadVariableOp(^QNetwork/dense_17/MatMul/ReadVariableOp*
T0	*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:?????????:?????????:?????????:?????????TT: : : : : : : : : : 2v
9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp2v
9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp2v
9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp2T
(QNetwork/dense_17/BiasAdd/ReadVariableOp(QNetwork/dense_17/BiasAdd/ReadVariableOp2R
'QNetwork/dense_17/MatMul/ReadVariableOp'QNetwork/dense_17/MatMul/ReadVariableOp:X T
#
_output_shapes
:?????????
-
_user_specified_nametime_step/step_type:UQ
#
_output_shapes
:?????????
*
_user_specified_nametime_step/reward:WS
#
_output_shapes
:?????????
,
_user_specified_nametime_step/discount:fb
/
_output_shapes
:?????????TT
/
_user_specified_nametime_step/observation
?
8
&__inference_get_initial_state_20465870

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?&
?
!__inference__traced_save_20466369
file_prefix'
#savev2_variable_read_readvariableop	H
Dsavev2_qnetwork_encodingnetwork_conv2d_24_kernel_read_readvariableopF
Bsavev2_qnetwork_encodingnetwork_conv2d_24_bias_read_readvariableopH
Dsavev2_qnetwork_encodingnetwork_conv2d_25_kernel_read_readvariableopF
Bsavev2_qnetwork_encodingnetwork_conv2d_25_bias_read_readvariableopH
Dsavev2_qnetwork_encodingnetwork_conv2d_26_kernel_read_readvariableopF
Bsavev2_qnetwork_encodingnetwork_conv2d_26_bias_read_readvariableopG
Csavev2_qnetwork_encodingnetwork_dense_16_kernel_read_readvariableopE
Asavev2_qnetwork_encodingnetwork_dense_16_bias_read_readvariableop7
3savev2_qnetwork_dense_17_kernel_read_readvariableop5
1savev2_qnetwork_dense_17_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableopDsavev2_qnetwork_encodingnetwork_conv2d_24_kernel_read_readvariableopBsavev2_qnetwork_encodingnetwork_conv2d_24_bias_read_readvariableopDsavev2_qnetwork_encodingnetwork_conv2d_25_kernel_read_readvariableopBsavev2_qnetwork_encodingnetwork_conv2d_25_bias_read_readvariableopDsavev2_qnetwork_encodingnetwork_conv2d_26_kernel_read_readvariableopBsavev2_qnetwork_encodingnetwork_conv2d_26_bias_read_readvariableopCsavev2_qnetwork_encodingnetwork_dense_16_kernel_read_readvariableopAsavev2_qnetwork_encodingnetwork_dense_16_bias_read_readvariableop3savev2_qnetwork_dense_17_kernel_read_readvariableop1savev2_qnetwork_dense_17_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapest
r: : : : : @:@:@@:@:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:&"
 
_output_shapes
:
??:!	

_output_shapes	
:?:%
!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: 
?
(
&__inference_signature_wrapper_20466308?
PartitionedCallPartitionedCall*	
Tin
 *

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_function_with_signature_204658942
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes 
?v
?
*__inference_polymorphic_action_fn_20465970
	step_type

reward
discount
observation[
Aqnetwork_encodingnetwork_conv2d_24_conv2d_readvariableop_resource: P
Bqnetwork_encodingnetwork_conv2d_24_biasadd_readvariableop_resource: [
Aqnetwork_encodingnetwork_conv2d_25_conv2d_readvariableop_resource: @P
Bqnetwork_encodingnetwork_conv2d_25_biasadd_readvariableop_resource:@[
Aqnetwork_encodingnetwork_conv2d_26_conv2d_readvariableop_resource:@@P
Bqnetwork_encodingnetwork_conv2d_26_biasadd_readvariableop_resource:@T
@qnetwork_encodingnetwork_dense_16_matmul_readvariableop_resource:
??P
Aqnetwork_encodingnetwork_dense_16_biasadd_readvariableop_resource:	?C
0qnetwork_dense_17_matmul_readvariableop_resource:	??
1qnetwork_dense_17_biasadd_readvariableop_resource:
identity	??9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp?9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp?9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp?8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp?(QNetwork/dense_17/BiasAdd/ReadVariableOp?'QNetwork/dense_17/MatMul/ReadVariableOp?
&QNetwork/EncodingNetwork/lambda_4/CastCastobservation*

DstT0*

SrcT0*/
_output_shapes
:?????????TT2(
&QNetwork/EncodingNetwork/lambda_4/Cast?
+QNetwork/EncodingNetwork/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2-
+QNetwork/EncodingNetwork/lambda_4/truediv/y?
)QNetwork/EncodingNetwork/lambda_4/truedivRealDiv*QNetwork/EncodingNetwork/lambda_4/Cast:y:04QNetwork/EncodingNetwork/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:?????????TT2+
)QNetwork/EncodingNetwork/lambda_4/truediv?
8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02:
8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_24/Conv2DConv2D-QNetwork/EncodingNetwork/lambda_4/truediv:z:0@QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2+
)QNetwork/EncodingNetwork/conv2d_24/Conv2D?
9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOpReadVariableOpBqnetwork_encodingnetwork_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp?
*QNetwork/EncodingNetwork/conv2d_24/BiasAddBiasAdd2QNetwork/EncodingNetwork/conv2d_24/Conv2D:output:0AQNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2,
*QNetwork/EncodingNetwork/conv2d_24/BiasAdd?
'QNetwork/EncodingNetwork/conv2d_24/ReluRelu3QNetwork/EncodingNetwork/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2)
'QNetwork/EncodingNetwork/conv2d_24/Relu?
8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02:
8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_25/Conv2DConv2D5QNetwork/EncodingNetwork/conv2d_24/Relu:activations:0@QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@*
paddingVALID*
strides
2+
)QNetwork/EncodingNetwork/conv2d_25/Conv2D?
9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOpReadVariableOpBqnetwork_encodingnetwork_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp?
*QNetwork/EncodingNetwork/conv2d_25/BiasAddBiasAdd2QNetwork/EncodingNetwork/conv2d_25/Conv2D:output:0AQNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@2,
*QNetwork/EncodingNetwork/conv2d_25/BiasAdd?
'QNetwork/EncodingNetwork/conv2d_25/ReluRelu3QNetwork/EncodingNetwork/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		@2)
'QNetwork/EncodingNetwork/conv2d_25/Relu?
8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02:
8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_26/Conv2DConv2D5QNetwork/EncodingNetwork/conv2d_25/Relu:activations:0@QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2+
)QNetwork/EncodingNetwork/conv2d_26/Conv2D?
9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOpReadVariableOpBqnetwork_encodingnetwork_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp?
*QNetwork/EncodingNetwork/conv2d_26/BiasAddBiasAdd2QNetwork/EncodingNetwork/conv2d_26/Conv2D:output:0AQNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2,
*QNetwork/EncodingNetwork/conv2d_26/BiasAdd?
'QNetwork/EncodingNetwork/conv2d_26/ReluRelu3QNetwork/EncodingNetwork/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2)
'QNetwork/EncodingNetwork/conv2d_26/Relu?
(QNetwork/EncodingNetwork/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2*
(QNetwork/EncodingNetwork/flatten_8/Const?
*QNetwork/EncodingNetwork/flatten_8/ReshapeReshape5QNetwork/EncodingNetwork/conv2d_26/Relu:activations:01QNetwork/EncodingNetwork/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2,
*QNetwork/EncodingNetwork/flatten_8/Reshape?
7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_16/MatMulMatMul3QNetwork/EncodingNetwork/flatten_8/Reshape:output:0?QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_16/MatMul?
8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_16/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_16/MatMul:product:0@QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_16/BiasAdd?
&QNetwork/EncodingNetwork/dense_16/ReluRelu2QNetwork/EncodingNetwork/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_16/Relu?
'QNetwork/dense_17/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'QNetwork/dense_17/MatMul/ReadVariableOp?
QNetwork/dense_17/MatMulMatMul4QNetwork/EncodingNetwork/dense_16/Relu:activations:0/QNetwork/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_17/MatMul?
(QNetwork/dense_17/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_17/BiasAdd/ReadVariableOp?
QNetwork/dense_17/BiasAddBiasAdd"QNetwork/dense_17/MatMul:product:00QNetwork/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_17/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_17/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMaxj
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShape"Categorical_1/mode/ArgMax:output:0*
T0	*
_output_shapes
:2
Deterministic_1/sample/Shape?
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastTo"Categorical_1/mode/ArgMax:output:0&Deterministic_1/sample/concat:output:0*
T0	*'
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0	*#
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:?????????2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:0:^QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp:^QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp:^QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp9^QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp)^QNetwork/dense_17/BiasAdd/ReadVariableOp(^QNetwork/dense_17/MatMul/ReadVariableOp*
T0	*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:?????????:?????????:?????????:?????????TT: : : : : : : : : : 2v
9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp2v
9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp2v
9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp2T
(QNetwork/dense_17/BiasAdd/ReadVariableOp(QNetwork/dense_17/BiasAdd/ReadVariableOp2R
'QNetwork/dense_17/MatMul/ReadVariableOp'QNetwork/dense_17/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	step_type:KG
#
_output_shapes
:?????????
 
_user_specified_namereward:MI
#
_output_shapes
:?????????
"
_user_specified_name
discount:\X
/
_output_shapes
:?????????TT
%
_user_specified_nameobservation
?
8
&__inference_get_initial_state_20466098

batch_size*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?5
?
$__inference__traced_restore_20466412
file_prefix#
assignvariableop_variable:	 V
<assignvariableop_1_qnetwork_encodingnetwork_conv2d_24_kernel: H
:assignvariableop_2_qnetwork_encodingnetwork_conv2d_24_bias: V
<assignvariableop_3_qnetwork_encodingnetwork_conv2d_25_kernel: @H
:assignvariableop_4_qnetwork_encodingnetwork_conv2d_25_bias:@V
<assignvariableop_5_qnetwork_encodingnetwork_conv2d_26_kernel:@@H
:assignvariableop_6_qnetwork_encodingnetwork_conv2d_26_bias:@O
;assignvariableop_7_qnetwork_encodingnetwork_dense_16_kernel:
??H
9assignvariableop_8_qnetwork_encodingnetwork_dense_16_bias:	?>
+assignvariableop_9_qnetwork_dense_17_kernel:	?8
*assignvariableop_10_qnetwork_dense_17_bias:
identity_12??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B%train_step/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/0/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/1/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/2/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/3/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/4/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/5/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/6/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/7/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/8/.ATTRIBUTES/VARIABLE_VALUEB,model_variables/9/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp<assignvariableop_1_qnetwork_encodingnetwork_conv2d_24_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp:assignvariableop_2_qnetwork_encodingnetwork_conv2d_24_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp<assignvariableop_3_qnetwork_encodingnetwork_conv2d_25_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp:assignvariableop_4_qnetwork_encodingnetwork_conv2d_25_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp<assignvariableop_5_qnetwork_encodingnetwork_conv2d_26_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp:assignvariableop_6_qnetwork_encodingnetwork_conv2d_26_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp;assignvariableop_7_qnetwork_encodingnetwork_dense_16_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp9assignvariableop_8_qnetwork_encodingnetwork_dense_16_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp+assignvariableop_9_qnetwork_dense_17_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp*assignvariableop_10_qnetwork_dense_17_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_11?
Identity_12IdentityIdentity_11:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_12"#
identity_12Identity_12:output:0*+
_input_shapes
: : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102(
AssignVariableOp_2AssignVariableOp_22(
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
?
l
,__inference_function_with_signature_20465883
unknown:	 
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *&
f!R
__inference_<lambda>_204656042
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
?
?
&__inference_signature_wrapper_20466291
discount
observation

reward
	step_type!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_function_with_signature_204658342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:?????????:?????????TT:?????????:?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:^Z
/
_output_shapes
:?????????TT
'
_user_specified_name0/observation:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:PL
#
_output_shapes
:?????????
%
_user_specified_name0/step_type
?
>
,__inference_function_with_signature_20465871

batch_size?
PartitionedCallPartitionedCall
batch_size*
Tin
2*

Tout
 *
_collective_manager_ids
 *
_output_shapes
 * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_get_initial_state_204658702
PartitionedCall*(
_construction_contextkEagerRuntime*
_input_shapes
: :B >

_output_shapes
: 
$
_user_specified_name
batch_size
?
?
,__inference_function_with_signature_20465834
	step_type

reward
discount
observation!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:
??
	unknown_6:	?
	unknown_7:	?
	unknown_8:
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2	*
_collective_manager_ids
 *#
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *3
f.R,
*__inference_polymorphic_action_fn_204658112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:?????????:?????????:?????????:?????????TT: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
#
_output_shapes
:?????????
%
_user_specified_name0/step_type:MI
#
_output_shapes
:?????????
"
_user_specified_name
0/reward:OK
#
_output_shapes
:?????????
$
_user_specified_name
0/discount:^Z
/
_output_shapes
:?????????TT
'
_user_specified_name0/observation
?
f
&__inference_signature_wrapper_20466304
unknown:	 
identity	??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallunknown*
Tin
2*
Tout
2	*
_collective_manager_ids
 *
_output_shapes
: *#
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *5
f0R.
,__inference_function_with_signature_204658832
StatefulPartitionedCall}
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0	*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 22
StatefulPartitionedCallStatefulPartitionedCall
?v
?
*__inference_polymorphic_action_fn_20465811
	time_step
time_step_1
time_step_2
time_step_3[
Aqnetwork_encodingnetwork_conv2d_24_conv2d_readvariableop_resource: P
Bqnetwork_encodingnetwork_conv2d_24_biasadd_readvariableop_resource: [
Aqnetwork_encodingnetwork_conv2d_25_conv2d_readvariableop_resource: @P
Bqnetwork_encodingnetwork_conv2d_25_biasadd_readvariableop_resource:@[
Aqnetwork_encodingnetwork_conv2d_26_conv2d_readvariableop_resource:@@P
Bqnetwork_encodingnetwork_conv2d_26_biasadd_readvariableop_resource:@T
@qnetwork_encodingnetwork_dense_16_matmul_readvariableop_resource:
??P
Aqnetwork_encodingnetwork_dense_16_biasadd_readvariableop_resource:	?C
0qnetwork_dense_17_matmul_readvariableop_resource:	??
1qnetwork_dense_17_biasadd_readvariableop_resource:
identity	??9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp?9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp?9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp?8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp?8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp?7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp?(QNetwork/dense_17/BiasAdd/ReadVariableOp?'QNetwork/dense_17/MatMul/ReadVariableOp?
&QNetwork/EncodingNetwork/lambda_4/CastCasttime_step_3*

DstT0*

SrcT0*/
_output_shapes
:?????????TT2(
&QNetwork/EncodingNetwork/lambda_4/Cast?
+QNetwork/EncodingNetwork/lambda_4/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2-
+QNetwork/EncodingNetwork/lambda_4/truediv/y?
)QNetwork/EncodingNetwork/lambda_4/truedivRealDiv*QNetwork/EncodingNetwork/lambda_4/Cast:y:04QNetwork/EncodingNetwork/lambda_4/truediv/y:output:0*
T0*/
_output_shapes
:?????????TT2+
)QNetwork/EncodingNetwork/lambda_4/truediv?
8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_24_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02:
8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_24/Conv2DConv2D-QNetwork/EncodingNetwork/lambda_4/truediv:z:0@QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2+
)QNetwork/EncodingNetwork/conv2d_24/Conv2D?
9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOpReadVariableOpBqnetwork_encodingnetwork_conv2d_24_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02;
9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp?
*QNetwork/EncodingNetwork/conv2d_24/BiasAddBiasAdd2QNetwork/EncodingNetwork/conv2d_24/Conv2D:output:0AQNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2,
*QNetwork/EncodingNetwork/conv2d_24/BiasAdd?
'QNetwork/EncodingNetwork/conv2d_24/ReluRelu3QNetwork/EncodingNetwork/conv2d_24/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2)
'QNetwork/EncodingNetwork/conv2d_24/Relu?
8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02:
8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_25/Conv2DConv2D5QNetwork/EncodingNetwork/conv2d_24/Relu:activations:0@QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@*
paddingVALID*
strides
2+
)QNetwork/EncodingNetwork/conv2d_25/Conv2D?
9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOpReadVariableOpBqnetwork_encodingnetwork_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp?
*QNetwork/EncodingNetwork/conv2d_25/BiasAddBiasAdd2QNetwork/EncodingNetwork/conv2d_25/Conv2D:output:0AQNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????		@2,
*QNetwork/EncodingNetwork/conv2d_25/BiasAdd?
'QNetwork/EncodingNetwork/conv2d_25/ReluRelu3QNetwork/EncodingNetwork/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:?????????		@2)
'QNetwork/EncodingNetwork/conv2d_25/Relu?
8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02:
8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp?
)QNetwork/EncodingNetwork/conv2d_26/Conv2DConv2D5QNetwork/EncodingNetwork/conv2d_25/Relu:activations:0@QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2+
)QNetwork/EncodingNetwork/conv2d_26/Conv2D?
9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOpReadVariableOpBqnetwork_encodingnetwork_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02;
9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp?
*QNetwork/EncodingNetwork/conv2d_26/BiasAddBiasAdd2QNetwork/EncodingNetwork/conv2d_26/Conv2D:output:0AQNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2,
*QNetwork/EncodingNetwork/conv2d_26/BiasAdd?
'QNetwork/EncodingNetwork/conv2d_26/ReluRelu3QNetwork/EncodingNetwork/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2)
'QNetwork/EncodingNetwork/conv2d_26/Relu?
(QNetwork/EncodingNetwork/flatten_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  2*
(QNetwork/EncodingNetwork/flatten_8/Const?
*QNetwork/EncodingNetwork/flatten_8/ReshapeReshape5QNetwork/EncodingNetwork/conv2d_26/Relu:activations:01QNetwork/EncodingNetwork/flatten_8/Const:output:0*
T0*(
_output_shapes
:??????????2,
*QNetwork/EncodingNetwork/flatten_8/Reshape?
7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOpReadVariableOp@qnetwork_encodingnetwork_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype029
7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp?
(QNetwork/EncodingNetwork/dense_16/MatMulMatMul3QNetwork/EncodingNetwork/flatten_8/Reshape:output:0?QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(QNetwork/EncodingNetwork/dense_16/MatMul?
8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOpReadVariableOpAqnetwork_encodingnetwork_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02:
8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp?
)QNetwork/EncodingNetwork/dense_16/BiasAddBiasAdd2QNetwork/EncodingNetwork/dense_16/MatMul:product:0@QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2+
)QNetwork/EncodingNetwork/dense_16/BiasAdd?
&QNetwork/EncodingNetwork/dense_16/ReluRelu2QNetwork/EncodingNetwork/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2(
&QNetwork/EncodingNetwork/dense_16/Relu?
'QNetwork/dense_17/MatMul/ReadVariableOpReadVariableOp0qnetwork_dense_17_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02)
'QNetwork/dense_17/MatMul/ReadVariableOp?
QNetwork/dense_17/MatMulMatMul4QNetwork/EncodingNetwork/dense_16/Relu:activations:0/QNetwork/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_17/MatMul?
(QNetwork/dense_17/BiasAdd/ReadVariableOpReadVariableOp1qnetwork_dense_17_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(QNetwork/dense_17/BiasAdd/ReadVariableOp?
QNetwork/dense_17/BiasAddBiasAdd"QNetwork/dense_17/MatMul:product:00QNetwork/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
QNetwork/dense_17/BiasAdd?
#Categorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
?????????2%
#Categorical_1/mode/ArgMax/dimension?
Categorical_1/mode/ArgMaxArgMax"QNetwork/dense_17/BiasAdd:output:0,Categorical_1/mode/ArgMax/dimension:output:0*
T0*#
_output_shapes
:?????????2
Categorical_1/mode/ArgMaxj
Deterministic/atolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
Deterministic/rtol?
#Deterministic_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB 2%
#Deterministic_1/sample/sample_shape?
Deterministic_1/sample/ShapeShape"Categorical_1/mode/ArgMax:output:0*
T0	*
_output_shapes
:2
Deterministic_1/sample/Shape?
'Deterministic_1/sample/BroadcastArgs/s1Const*
_output_shapes
: *
dtype0*
valueB 2)
'Deterministic_1/sample/BroadcastArgs/s1?
$Deterministic_1/sample/BroadcastArgsBroadcastArgs%Deterministic_1/sample/Shape:output:00Deterministic_1/sample/BroadcastArgs/s1:output:0*
_output_shapes
:2&
$Deterministic_1/sample/BroadcastArgs
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/Const?
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0?
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis?
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0)Deterministic_1/sample/BroadcastArgs:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat?
"Deterministic_1/sample/BroadcastToBroadcastTo"Categorical_1/mode/ArgMax:output:0&Deterministic_1/sample/concat:output:0*
T0	*'
_output_shapes
:?????????2$
"Deterministic_1/sample/BroadcastTo?
Deterministic_1/sample/Shape_1Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0	*
_output_shapes
:2 
Deterministic_1/sample/Shape_1?
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack?
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1?
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2?
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_1:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_slice?
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axis?
Deterministic_1/sample/concat_1ConcatV2,Deterministic_1/sample/sample_shape:output:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1?
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0	*#
_output_shapes
:?????????2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2
clip_by_value/Minimum/y?
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0	*#
_output_shapes
:?????????2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
clip_by_value/y?
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0	*#
_output_shapes
:?????????2
clip_by_value?
IdentityIdentityclip_by_value:z:0:^QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp:^QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp:^QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp9^QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp9^QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp8^QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp)^QNetwork/dense_17/BiasAdd/ReadVariableOp(^QNetwork/dense_17/MatMul/ReadVariableOp*
T0	*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*o
_input_shapes^
\:?????????:?????????:?????????:?????????TT: : : : : : : : : : 2v
9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp9QNetwork/EncodingNetwork/conv2d_24/BiasAdd/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_24/Conv2D/ReadVariableOp2v
9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp9QNetwork/EncodingNetwork/conv2d_25/BiasAdd/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_25/Conv2D/ReadVariableOp2v
9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp9QNetwork/EncodingNetwork/conv2d_26/BiasAdd/ReadVariableOp2t
8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp8QNetwork/EncodingNetwork/conv2d_26/Conv2D/ReadVariableOp2t
8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp8QNetwork/EncodingNetwork/dense_16/BiasAdd/ReadVariableOp2r
7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp7QNetwork/EncodingNetwork/dense_16/MatMul/ReadVariableOp2T
(QNetwork/dense_17/BiasAdd/ReadVariableOp(QNetwork/dense_17/BiasAdd/ReadVariableOp2R
'QNetwork/dense_17/MatMul/ReadVariableOp'QNetwork/dense_17/MatMul/ReadVariableOp:N J
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:NJ
#
_output_shapes
:?????????
#
_user_specified_name	time_step:ZV
/
_output_shapes
:?????????TT
#
_user_specified_name	time_step"?L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
action?
4

0/discount&
action_0/discount:0?????????
F
0/observation5
action_0/observation:0?????????TT
0
0/reward$
action_0/reward:0?????????
6
0/step_type'
action_0/step_type:0?????????6
action,
StatefulPartitionedCall:0	?????????tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*,
get_metadatatensorflow/serving/predict*Z
get_train_stepH*
int64!
StatefulPartitionedCall_1:0	 tensorflow/serving/predict:??
?

train_step
metadata
model_variables
_all_assets

signatures

naction
odistribution
pget_initial_state
qget_metadata
rget_train_step"
_generic_user_object
:	 (2Variable
 "
trackable_dict_wrapper
g
0
1
2
	3

4
5
6
7
8
9"
trackable_tuple_wrapper
'
0"
trackable_list_wrapper
`

saction
tget_initial_state
uget_train_step
vget_metadata"
signature_map
C:A 2)QNetwork/EncodingNetwork/conv2d_24/kernel
5:3 2'QNetwork/EncodingNetwork/conv2d_24/bias
C:A @2)QNetwork/EncodingNetwork/conv2d_25/kernel
5:3@2'QNetwork/EncodingNetwork/conv2d_25/bias
C:A@@2)QNetwork/EncodingNetwork/conv2d_26/kernel
5:3@2'QNetwork/EncodingNetwork/conv2d_26/bias
<::
??2(QNetwork/EncodingNetwork/dense_16/kernel
5:3?2&QNetwork/EncodingNetwork/dense_16/bias
+:)	?2QNetwork/dense_17/kernel
$:"2QNetwork/dense_17/bias
1
ref
1"
trackable_tuple_wrapper
.

_q_network"
_generic_user_object
?
_encoder
_q_value_layer
regularization_losses
trainable_variables
	variables
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"?
_tf_keras_layer?{"name": "QNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QNetwork", "config": {"layer was saved without config": true}}
?
_flat_preprocessing_layers
_postprocessing_layers
regularization_losses
trainable_variables
	variables
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"?
_tf_keras_layer?{"name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "EncodingNetwork", "config": {"layer was saved without config": true}}
?

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
*{&call_and_return_all_conditional_losses
|__call__"?
_tf_keras_layer?{"name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 512]}}
 "
trackable_list_wrapper
f
0
1
2
	3

4
5
6
7
8
9"
trackable_list_wrapper
f
0
1
2
	3

4
5
6
7
8
9"
trackable_list_wrapper
?
#metrics
$layer_metrics

%layers
regularization_losses
trainable_variables
	variables
&non_trainable_variables
'layer_regularization_losses
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
'
(0"
trackable_list_wrapper
C
)0
*1
+2
,3
-4"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
	3

4
5
6
7"
trackable_list_wrapper
X
0
1
2
	3

4
5
6
7"
trackable_list_wrapper
?
.metrics
/layer_metrics

0layers
regularization_losses
trainable_variables
	variables
1non_trainable_variables
2layer_regularization_losses
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
3metrics
4layer_metrics

5layers
regularization_losses
 trainable_variables
!	variables
6non_trainable_variables
7layer_regularization_losses
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
8regularization_losses
9trainable_variables
:	variables
;	keras_api
*}&call_and_return_all_conditional_losses
~__call__"?
_tf_keras_layer?{"name": "lambda_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_4", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAQwAAAHMSAAAAdACgAXwAdAJqA6ECZAEbAFMAKQJOZwAAAAAA\n4G9AKQTaAnRm2gRjYXN02gJucNoHZmxvYXQzMikB2gNvYnOpAHIGAAAAeiA8aXB5dGhvbi1pbnB1\ndC0xNjktY2RiNzk3MzBkOTE2PtoIPGxhbWJkYT4EAAAA8wAAAAA=\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
?


kernel
bias
<regularization_losses
=trainable_variables
>	variables
?	keras_api
*&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 84, 84, 4]}}
?


kernel
	bias
@regularization_losses
Atrainable_variables
B	variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 20, 20, 32]}}
?



kernel
bias
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 9, 9, 64]}}
?
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatten_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_8", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

kernel
bias
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3136}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 3136]}}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
(0
)1
*2
+3
,4
-5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Pmetrics
Qlayer_metrics

Rlayers
8regularization_losses
9trainable_variables
:	variables
Snon_trainable_variables
Tlayer_regularization_losses
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Umetrics
Vlayer_metrics

Wlayers
<regularization_losses
=trainable_variables
>	variables
Xnon_trainable_variables
Ylayer_regularization_losses
?__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
?
Zmetrics
[layer_metrics

\layers
@regularization_losses
Atrainable_variables
B	variables
]non_trainable_variables
^layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
?
_metrics
`layer_metrics

alayers
Dregularization_losses
Etrainable_variables
F	variables
bnon_trainable_variables
clayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
dmetrics
elayer_metrics

flayers
Hregularization_losses
Itrainable_variables
J	variables
gnon_trainable_variables
hlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
imetrics
jlayer_metrics

klayers
Lregularization_losses
Mtrainable_variables
N	variables
lnon_trainable_variables
mlayer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
*__inference_polymorphic_action_fn_20465970
*__inference_polymorphic_action_fn_20466041?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_polymorphic_distribution_fn_20466095?
???
FullArgSpec(
args ?
j	time_step
jpolicy_state
varargs
 
varkw
 
defaults?
? 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_get_initial_state_20466098?
???
FullArgSpec!
args?
jself
j
batch_size
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
__inference_<lambda>_20465607"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
__inference_<lambda>_20465604"?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_20466291
0/discount0/observation0/reward0/step_type"?
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
?B?
&__inference_signature_wrapper_20466296
batch_size"?
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
?B?
&__inference_signature_wrapper_20466304"?
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
?B?
&__inference_signature_wrapper_20466308"?
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
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpecL
argsD?A
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaults?

 
? 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
 <
__inference_<lambda>_20465604?

? 
? "? 	5
__inference_<lambda>_20465607?

? 
? "? S
&__inference_get_initial_state_20466098)"?
?
?

batch_size 
? "? ?
*__inference_polymorphic_action_fn_20465970?
	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount?????????<
observation-?*
observation?????????TT
? 
? "R?O

PolicyStep&
action?
action?????????	
state? 
info? ?
*__inference_polymorphic_action_fn_20466041?
	
???
???
???
TimeStep6
	step_type)?&
time_step/step_type?????????0
reward&?#
time_step/reward?????????4
discount(?%
time_step/discount?????????F
observation7?4
time_step/observation?????????TT
? 
? "R?O

PolicyStep&
action?
action?????????	
state? 
info? ?
0__inference_polymorphic_distribution_fn_20466095?
	
???
???
???
TimeStep,
	step_type?
	step_type?????????&
reward?
reward?????????*
discount?
discount?????????<
observation-?*
observation?????????TT
? 
? "???

PolicyStep?
action?????Ã}?z
`
C?@
"j tf_agents.policies.greedy_policy
jDeterministicWithLogProb
*?'
%
loc?
Identity?????????	
? _TFPTypeSpec
state? 
info? ?
&__inference_signature_wrapper_20466291?
	
???
? 
???
.

0/discount ?

0/discount?????????
@
0/observation/?,
0/observation?????????TT
*
0/reward?
0/reward?????????
0
0/step_type!?
0/step_type?????????"+?(
&
action?
action?????????	a
&__inference_signature_wrapper_2046629670?-
? 
&?#
!

batch_size?

batch_size "? Z
&__inference_signature_wrapper_204663040?

? 
? "?

int64?
int64 	>
&__inference_signature_wrapper_20466308?

? 
? "? 