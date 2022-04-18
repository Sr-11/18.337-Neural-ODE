using DiffEqFlux, DifferentialEquations, Plots, GalacticOptim

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.0f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= true_A*u
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = FastChain(FastDense(2, 50, tanh),
                  FastDense(50, 2))
prob_neuralode = NeuralODE(dudt2, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

callback = function (p, l, pred; doplot = true)
  display(l)
  # plot current prediction against data
  plt = scatter(tsteps, ode_data[1,:], label = "data1")
  scatter!(plt, tsteps, pred[1,:], label = "prediction1")
  scatter!(plt, tsteps, pred[2,:], label = "data2")
  scatter!(plt, tsteps, pred[2,:], label = "prediction2")
  if doplot
    display(plot(plt))
  end
  return false
end

# use GalacticOptim.jl to solve the problem
adtype = GalacticOptim.AutoZygote()

optf = GalacticOptim.OptimizationFunction((x, p) -> loss_neuralode(x), adtype)
optfunc = GalacticOptim.instantiate_function(optf, prob_neuralode.p, adtype, nothing)
optprob = GalacticOptim.OptimizationProblem(optfunc, prob_neuralode.p)

result_neuralode = GalacticOptim.solve(optprob,
                                       ADAM(0.05),
                                       cb = callback,
                                       maxiters = 300)