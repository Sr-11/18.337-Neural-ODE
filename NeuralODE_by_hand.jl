########## Part 2: pullbackpropagation of a neural network ##########
using Plots
u=rand(2,1)
W1=rand(50,2);W2=rand(2,50);b1=rand(50,1);b2=rand(2,1)
y=rand(2,1)
u⁻=similar(u)
W1⁻=similar(W1);W2⁻=similar(W2);b1⁻=similar(b1);b2⁻=similar(b2)
y⁻=similar(y)

function forward!(y,u,W1,W2,b1,b2)
  y.=W2*tanh.(W1*u+b1).+b2
end
function pullback!(u⁻,W1⁻,W2⁻,b1⁻,b2⁻,u,W1,W2,b1,b2,y⁻)
  b2⁻.=y⁻
  W2⁻.=y⁻*tanh.(W1*u+b1)'
  W1⁻.=(W2'*y⁻./(cosh.(W1*u+b1)).^2)*u'
  b1⁻.=W2'*y⁻./(cosh.(W1*u+b1)).^2
  u⁻.=W1'*b1⁻
  return nothing
end
y⁻.=[2.0;1.0]
forward!(y,u,W1,W2,b1,b2)
pullback!(u⁻,W1⁻,W2⁻,b1⁻,b2⁻,u,W1,W2,b1,b2,y⁻)
using ForwardDiff
function NN_ForwardDiff(u,W1,W2,b1,b2)
  return W2*tanh.(W1*u+b1)+b2
end
Jacobianᵘ=ForwardDiff.jacobian(u->NN_ForwardDiff(u,W1,W2,b1,b2), u)
Jacobianᵂ¹=ForwardDiff.jacobian(W1->NN_ForwardDiff(u,W1,W2,b1,b2), W1)
Jacobianᵂ²=ForwardDiff.jacobian(W2->NN_ForwardDiff(u,W1,W2,b1,b2), W2)
Jacobianᵇ¹=ForwardDiff.jacobian(b1->NN_ForwardDiff(u,W1,W2,b1,b2), b1)
Jacobianᵇ²=ForwardDiff.jacobian(b2->NN_ForwardDiff(u,W1,W2,b1,b2), b2)
Bₙₙᵘ=Jacobianᵘ'*y⁻
Bₙₙᵂ¹=[Jacobianᵂ¹[1:2,1:50]'*y⁻ Jacobianᵂ¹[1:2,51:100]'*y⁻]
Bₙₙᵂ²=[y⁻'*Jacobianᵂ²[1:2,1:2:100];y⁻'*Jacobianᵂ²[1:2,2:2:100]]
Bₙₙᵇ¹=(Jacobianᵇ¹'*y⁻)[:,:]
Bₙₙᵇ²=(Jacobianᵇ²'*y⁻)[:,:]

print_deci_3(x)=println(round.(x; digits=3))
println("########## Part 2: pullbackpropagation of a neural network ##########")
println("My pull pullback Bₙₙᵘ=")
print_deci_3(u⁻)
println("ForwardDiff output Bₙₙᵘ=")
print_deci_3(Bₙₙᵘ)
print('\n')

println("My pull pullback Bₙₙᵂ¹=")
print_deci_3(W1⁻)
println("ForwardDiff output Bₙₙᵂ¹=")
print_deci_3(Bₙₙᵂ¹)
print('\n')

println("My pull pullback Bₙₙᵂ²=")
print_deci_3(W2⁻)
println("ForwardDiff output Bₙₙᵂ²=")
print_deci_3(Bₙₙᵂ²)
print('\n')

println("My pull pullback Bₙₙᵇ¹=")
print_deci_3(b1⁻)
println("ForwardDiff output Bₙₙᵇ¹=")
print_deci_3(Bₙₙᵇ¹)
print('\n')

println("My pull pullback Bₙₙᵇ²=")
print_deci_3(b2⁻)
println("ForwardDiff output Bₙₙᵇ²=")
print_deci_3(Bₙₙᵇ²)
print('\n')

########## Part 3: Implementing an ODE adjoint ##########
using DifferentialEquations
########## Get the data from the Spiral ODE example ##########
u0 = [2.0; 0.0;;]
tspan = (0.0, 1.0)
tsteps = 0:0.1:1
function trueODEfunc(du, u, p, t)
  true_A = [-0.1 2.0; -2.0 -0.1]
  du .= true_A*u
end
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
sol_trueode=solve(prob_trueode, Tsit5(), saveat = tsteps)
########## Define the forward ODE of u ###########
function assign!(W1,W2,b1,b2,p)
  for i in 1:100 W1[i]=p[i] end
  for i in 1:100 W2[i]=p[i+100] end
  for i in 1:50 b1[i]=p[i+200] end
  for i in 1:2 b2[i]=p[i+250] end
end
function flatten!(p,W1,W2,b1,b2)
  for i in 1:100 p[i]=W1[i] end
  for i in 1:100 p[i+100]=W2[i] end
  for i in 1:50 p[i+200]=b1[i] end
  for i in 1:2 p[i+250]=b2[i] end
end
function ODE_forward(p)
  W1=rand(50,2);W2=rand(2,50);b1=rand(50,1);b2=rand(2,1)
  assign!(W1,W2,b1,b2,p)
  function dudt!(du,u,p,t)
    forward!(du,u,W1,W2,b1,b2)
  end
  prob_node = ODEProblem(dudt!, u0, tspan, p)
  sol_node = solve(prob_node, Tsit5())
  return prob_node,sol_node
end
function p_reset()
  b1=zeros(50,1);b2=zeros(2,1)
  ϵ=0.0001
  W1=zeros(50,2);W2=zeros(2,50);
  W1[1,1]=1;W1[2,2]=1;
  W2[1,1]=-0.1;W2[1,2]=2;W2[2,1]=-2;W2[2,2]=-0.1
  W1=W1.*ϵ
  W2=W2./ϵ
  flatten!(p,W1,W2,b1,b2)
end
p_reset()
prob_node,sol_node=ODE_forward(p)
########## Implementing inverse ODE of λ and μ  ##########
########## Denote λμ=[λ;μ] ##########
########## λ=λμ[1:2,:] ##########
########## μ=λμ[3:254,:] ##########
function ODE_adjoint(prob_node,sol_node)
  p=prob_node.p
  W1=rand(50,2);W2=rand(2,50);b1=rand(50,1);b2=rand(2,1);
  assign!(W1,W2,b1,b2,p)
  function dλμdt!(dλμ,λμ,p,t)
    y=rand(2,1)
    W1⁻=similar(W1);W2⁻=similar(W2);b1⁻=similar(b1);b2⁻=similar(b2);
    u⁻=similar(rand(2,1));p⁻=similar(rand(252,1))
    forward!(y,sol_node(t),W1,W2,b1,b2)
    pullback!(u⁻,W1⁻,W2⁻,b1⁻,b2⁻,sol_node(t),W1,W2,b1,b2,λμ[1:2,:])
    flatten!(p⁻,W1⁻,W2⁻,b1⁻,b2⁻)
    dλμ.=-[u⁻;p⁻]
  end
  tspan_λμ=(tspan[2],tspan[1])
  step=0.1
  tsteps_λμ=0:step:1
  λμT=[2*(sol_node(1)-sol_trueode(1))*(tsteps_λμ[2]-tsteps_λμ[1]); zeros(252)]
  dosetimes=collect(tsteps_λμ)
  condition(u,t,integrator) = t ∈ dosetimes
  function affect!(integrator)
    integrator.u[1:2] .+= 2*(sol_node(integrator.t)-sol_trueode(integrator.t))*(tsteps_λμ[2]-tsteps_λμ[1])
  end
  cb = DiscreteCallback(condition,affect!)

  prob_inverse = ODEProblem(dλμdt!, λμT, tspan_λμ, p)
  sol_inverse = solve(prob_inverse, Tsit5(), callback=cb, tstops=dosetimes)
  return sol_inverse(0)[3:254]
  #using Plots
  #pltλ=scatter(tsteps, [ui[1] for ui in sol_inverse.u], label = "λ1")
  #scatter!(pltλ,tsteps, [ui[2] for ui in sol_inverse.u], label = "λ2")
end

########## Numerical Check ##########
function loss(sol_node)
  Δ=[sol_node(t) - sol_trueode(t) for t in 0:0.01:1]
  D=[sum(abs2,δ) for δ in Δ]
  return sum(D)/length(D)
end
function my_test()
  p=rand(252)
  prob_node,sol_node=ODE_forward(p);
  dCdp=ODE_adjoint(prob_node,sol_node)
  ϵ=0.0000001
  Δp=ϵ*rand(252)
  prob_node_ad,sol_node_ad=ODE_forward(p+Δp);
  println("Should be: ")
  println(loss(sol_node_ad)-loss(sol_node))
  #print_deci_3(loss(sol_node_ad)-loss(sol_node)/Δp[1])
  println("My adjoint: ")
  println(sum(dCdp.*Δp))
  #print_deci_3(dCdp[1])
end
my_test()
########## Part 4: Training the neural ODE ##########

########## Plot callback ##########

function cb(tmp_sol_node)
  tsteps=0:0.01:1
  plt = scatter(tsteps, [sol_trueode(t)[1] for t in tsteps], label = "data_u1")
  scatter!(plt, tsteps, [sol_trueode(t)[2] for t in tsteps], label = "data_u2")
  scatter!(plt, tsteps, [tmp_sol_node(t)[1] for t in tsteps], label = "pred_u1")
  scatter!(plt, tsteps, [tmp_sol_node(t)[2] for t in tsteps], label = "pred_u2")
  display(plt)
  println(loss(tmp_sol_node))
end
function train(;p,η,iters)
  p_new=copy(p)
  for i in 1:iters
    prob_node,sol_node=ODE_forward(p)
    dCdp=ODE_adjoint(prob_node,sol_node)
    p_new.=p.-η.*dCdp
    if sum(abs2,p_new-p)==0
      break
    end
    p.=p_new
    cb(sol_node)
  end
end
using BenchmarkTools
train(p=p,η=0.01,iters=1000)

##### Answer #####
