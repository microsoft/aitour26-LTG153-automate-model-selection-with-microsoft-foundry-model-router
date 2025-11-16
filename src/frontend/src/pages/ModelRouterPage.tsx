import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Separator } from '@/components/ui/separator'
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '@/components/ui/collapsible'
import { useToast } from '@/hooks/use-toast'
import { getAuthHeaders } from '@/utils/auth'
import { modelRouterApi, Scenario, ModelResponse, ComparisonResponse, PricingData, AccuracyComparisonResponse, createApiWrapper } from '@/utils/api/apiWrapper'
import { LatencyComparison } from '@/components/LatencyComparison'
import { AccuracyComparison } from '@/components/AccuracyComparison'
import { Clock, DollarSign, Zap, BarChart3, Sparkles, Bot, Cpu, Target, ChevronDown } from 'lucide-react'
import replayData from '@/data/replayData.json'

const ROUTER_CLASSIFICATION_INPUT_RATE = 0.14

const departments = [
  { value: 'Finance', label: 'Finance', icon: DollarSign, color: 'bg-green-100 text-green-800' },
  { value: 'Marketing', label: 'Marketing', icon: BarChart3, color: 'bg-blue-100 text-blue-800' },
  { value: 'Development', label: 'Development', icon: Cpu, color: 'bg-purple-100 text-purple-800' }
]

const complexityColors = {
  Low: 'bg-green-100 text-green-800',
  Medium: 'bg-yellow-100 text-yellow-800',
  High: 'bg-red-100 text-red-800'
}

interface CostAnalysis {
  inputCost: number
  outputCost: number
  totalCost: number
  pricing: { input_per_1m: number; output_per_1m: number }
  usedFallback: boolean
  classificationCost: number
}

function calculateCost(
  promptTokens: number,
  completionTokens: number,
  modelType: string,
  pricing: PricingData['models'],
  includeRouterSurcharge = false
): CostAnalysis | null {
  let modelPricing = pricing[modelType]
  let usedFallback = false

  if (!modelPricing && pricing['default']) {
    modelPricing = pricing['default']
    usedFallback = true
  }

  if (!modelPricing) {
    return null
  }

  const inputCost = (promptTokens / 1000000) * modelPricing.input_per_1m
  const outputCost = (completionTokens / 1000000) * modelPricing.output_per_1m
  const classificationCost = includeRouterSurcharge ? (promptTokens / 1000000) * ROUTER_CLASSIFICATION_INPUT_RATE : 0

  return {
    inputCost,
    outputCost,
    classificationCost,
    totalCost: inputCost + outputCost + classificationCost,
    pricing: modelPricing,
    usedFallback
  }
}

function ResponseCard({ 
  title, 
  icon: Icon, 
  isLoading, 
  result, 
  pricing,
  isDark = false,
  includeRouterSurcharge = false
}: { 
  title: string
  icon: React.ElementType
  isLoading: boolean
  result: ModelResponse | null
  pricing: PricingData['models']
  isDark?: boolean
  includeRouterSurcharge?: boolean
}) {
  const cost = result ? calculateCost(
    result.prompt_tokens,
    result.completion_tokens,
    result.model_type,
    pricing,
    includeRouterSurcharge
  ) : null

  return (
    <Card className={`h-full ${isDark ? 'bg-slate-900 border-slate-700' : ''}`}>
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Icon className={`h-5 w-5 ${isDark ? 'text-slate-300' : 'text-slate-600'}`} />
          <CardTitle className={`text-lg ${isDark ? 'text-white' : ''}`}>{title}</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading ? (
          <div className="space-y-3">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-20 w-full" />
          </div>
        ) : result ? (
          <>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className={`text-sm font-medium ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>Model:</span>
                <Badge variant="outline" className={isDark ? 'border-slate-600 text-slate-300' : ''}>
                  {result.model_type}
                </Badge>
              </div>
              <div className="flex justify-between items-center">
                <span className={`text-sm font-medium ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>Tokens:</span>
                <div className="text-sm">
                  <span className={isDark ? 'text-slate-300' : 'text-slate-600'}>
                    {result.prompt_tokens} / {result.completion_tokens} / {result.total_tokens}
                  </span>
                </div>
              </div>
              {/* Latency Metrics */}
              {result.server_processing_ms && (
                <div className="flex justify-between items-center">
                  <span className={`text-sm font-medium ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>Server:</span>
                  <div className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    <span className="text-sm">{result.server_processing_ms}ms</span>
                  </div>
                </div>
              )}
              {result.network_ms !== undefined && (
                <div className="flex justify-between items-center">
                  <span className={`text-sm font-medium ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>Network:</span>
                  <div className="flex items-center gap-1">
                    <Zap className="h-3 w-3" />
                    <span className="text-sm">{result.network_ms.toFixed(1)}ms</span>
                  </div>
                </div>
              )}
              {result.response_time_ms && (
                <div className="flex justify-between items-center">
                  <span className={`text-sm font-medium ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>Total:</span>
                  <div className="flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    <span className="text-sm font-bold">{result.response_time_ms.toFixed(1)}ms</span>
                  </div>
                </div>
              )}
              {result && !cost && (
                <Alert className={`bg-amber-50 border-amber-200 ${isDark ? 'bg-opacity-20 border-amber-400' : ''}`}>
                  <AlertDescription className={`${isDark ? 'text-amber-300' : 'text-amber-800'}`}>
                    Pricing data for <strong>{result.model_type}</strong> is missing. The router likely selected a newer Azure model that this demo has not mapped yet. Add it to <code className="px-1 bg-amber-100 rounded">pricing.json</code> so costs stay accurate.
                  </AlertDescription>
                </Alert>
              )}
              {cost?.usedFallback && (
                <Alert className={`bg-amber-50 border-amber-200 ${isDark ? 'bg-opacity-20 border-amber-400' : ''}`}>
                  <AlertDescription className={`${isDark ? 'text-amber-300' : 'text-amber-800'}`}>
                    Using fallback pricing because <strong>{result?.model_type}</strong> is not mapped yet. Azure may have routed to a newly onboarded model; update <code className="px-1 bg-amber-100 rounded">pricing.json</code> to capture its exact rates. Fallback input/output: ${cost.pricing.input_per_1m.toFixed(2)} / ${cost.pricing.output_per_1m.toFixed(2)} (USD per 1M tokens).
                  </AlertDescription>
                </Alert>
              )}
              {cost && (
                <div className="space-y-1 pt-2 border-t">
                  <div className="flex justify-between items-center">
                    <span className={`text-sm font-medium ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>Cost:</span>
                    <span className="text-sm font-bold text-green-600">
                      ${cost.totalCost.toFixed(6)}
                    </span>
                  </div>
                  <div className={`text-xs ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                    Input: ${cost.inputCost.toFixed(6)} | Output: ${cost.outputCost.toFixed(6)}
                    {cost.classificationCost > 0 && (
                      <div className="mt-1">
                        Router classification: ${cost.classificationCost.toFixed(6)}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
            <Separator />
            <div className="space-y-2">
              <h4 className={`font-medium ${isDark ? 'text-white' : ''}`}>Response:</h4>
              <div className={`p-3 rounded-md text-sm leading-relaxed max-h-40 overflow-y-auto ${
                isDark ? 'bg-slate-800 text-slate-200' : 'bg-slate-50'
              }`}>
                {result.output}
              </div>
            </div>
          </>
        ) : (
          <div className={`text-center py-8 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
            <Icon className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>Awaiting prompt submission...</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default function ModelRouterPage() {
  const { toast } = useToast()
  const apiWrapper = createApiWrapper(toast)

  const [offlineMode, setOfflineMode] = useState(false)
  const [selectedDepartment, setSelectedDepartment] = useState<string>('Finance')
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [selectedScenario, setSelectedScenario] = useState<Scenario | null>(null)
  const [prompt, setPrompt] = useState('')
  const [groundTruth, setGroundTruth] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [routerLoading, setRouterLoading] = useState(false)
  const [benchmarkLoading, setBenchmarkLoading] = useState(false)
  const [routerResult, setRouterResult] = useState<ModelResponse | null>(null)
  const [benchmarkResult, setBenchmarkResult] = useState<ModelResponse | null>(null)
  const [pricing, setPricing] = useState<PricingData['models']>({})
  const [scenariosLoading, setScenariosLoading] = useState(false)
  const [accuracyComparison, setAccuracyComparison] = useState<AccuracyComparisonResponse | null>(null)
  const [accuracyLoading, setAccuracyLoading] = useState(false)

  // Check for offline mode
  useEffect(() => {
    const checkOfflineMode = () => {
      const isOffline = sessionStorage.getItem('offlineMode') === 'true'
      setOfflineMode(isOffline)
    }
    checkOfflineMode()
    const interval = setInterval(checkOfflineMode, 1000)
    return () => clearInterval(interval)
  }, [])

  // Load scenarios when department changes
  useEffect(() => {
    const loadScenarios = async () => {
      if (offlineMode) {
        // In offline mode, load from replay data
        const modelRouterData = replayData.modelRouterScenarios as any
        const departmentData = modelRouterData[selectedDepartment]
        
        if (departmentData) {
          const offlineScenarios: Scenario[] = Object.entries(departmentData).map(([id, data]: [string, any]) => ({
            id,
            title: data.title,
            prompt: data.prompt,
            complexity: data.complexity,
            qualityExpectation: data.qualityExpectation
          }))
          setScenarios(offlineScenarios)
          setSelectedScenario(null)
          setPrompt('')
        } else {
          setScenarios([])
        }
      } else {
        // In live mode, fetch from API
        try {
          await apiWrapper(
            () => modelRouterApi.getScenarios(selectedDepartment),
            {
              loadingState: [scenariosLoading, setScenariosLoading],
              onSuccess: (data) => {
                setScenarios(data)
                setSelectedScenario(null)
                setPrompt('')
              },
              successMessage: '',
              errorMessage: 'Failed to load scenarios'
            }
          )
        } catch (error) {
          console.error('Failed to load scenarios:', error)
        }
      }
    }
    loadScenarios()
  }, [selectedDepartment, offlineMode])

  // Load pricing on mount
  useEffect(() => {
    const loadPricing = async () => {
      if (offlineMode) {
        // In offline mode, use hardcoded pricing data
        setPricing({
          'gpt-5-mini-2025-08-07': {
            input_per_1m: 0.25,
            output_per_1m: 2.00,
            description: 'GPT-5 Mini - Fast and efficient'
          },
          'gpt-5-2025-08-07': {
            input_per_1m: 1.25,
            output_per_1m: 10.00,
            description: 'GPT-5 - Most capable model'
          },
          'default': {
            input_per_1m: 0.50,
            output_per_1m: 2.50,
            description: 'Default fallback pricing'
          }
        })
      } else {
        try {
          const data = await apiWrapper(
            () => modelRouterApi.getPricing(),
            {
              onSuccess: (data) => setPricing(data.models),
              successMessage: '',
              errorMessage: 'Failed to load pricing data'
            }
          )
        } catch (error) {
          console.error('Failed to load pricing:', error)
        }
      }
    }
    loadPricing()
  }, [offlineMode])

  const handleScenarioSelect = (scenario: Scenario) => {
    setSelectedScenario(scenario)
    setPrompt(scenario.prompt)
    setGroundTruth('')
    // Clear previous results
    setRouterResult(null)
    setBenchmarkResult(null)
    setAccuracyComparison(null)
  }

  const handleRunAccuracyComparison = async () => {
    if (!routerResult || !benchmarkResult) return
    
    setAccuracyLoading(true)
    
    // Check if offline mode and we have replay data
    if (offlineMode && selectedScenario) {
      const modelRouterData = replayData.modelRouterScenarios as any
      const departmentData = modelRouterData[selectedDepartment]
      const scenarioData = departmentData?.[selectedScenario.id]
      
      if (scenarioData?.accuracy) {
        setTimeout(() => {
          setAccuracyComparison({
            scenario_id: selectedScenario.id,
            router: {
              model_type: scenarioData.router.model,
              output: scenarioData.router.output,
              prompt_tokens: scenarioData.router.promptTokens,
              completion_tokens: scenarioData.router.completionTokens,
              total_tokens: scenarioData.router.totalTokens,
              accuracy_evaluation: {
                score: scenarioData.accuracy.router.score,
                reasoning: scenarioData.accuracy.router.reasoning,
                strengths: scenarioData.accuracy.router.strengths,
                weaknesses: scenarioData.accuracy.router.weaknesses,
                model_evaluated: scenarioData.router.model
              }
            },
            benchmark: {
              model_type: scenarioData.benchmark.model,
              output: scenarioData.benchmark.output,
              prompt_tokens: scenarioData.benchmark.promptTokens,
              completion_tokens: scenarioData.benchmark.completionTokens,
              total_tokens: scenarioData.benchmark.totalTokens,
              accuracy_evaluation: {
                score: scenarioData.accuracy.benchmark.score,
                reasoning: scenarioData.accuracy.benchmark.reasoning,
                strengths: scenarioData.accuracy.benchmark.strengths,
                weaknesses: scenarioData.accuracy.benchmark.weaknesses,
                model_evaluated: scenarioData.benchmark.model
              }
            },
            timing: {
              response_generation_ms: scenarioData.benchmark.responseTimeMs,
              accuracy_evaluation_ms: 500,
              total_ms: scenarioData.benchmark.responseTimeMs + 500
            }
          })
          setAccuracyLoading(false)
        }, 500)
        return
      }
    }
    
    // Live mode: call API
    await apiWrapper(
      () => modelRouterApi.accuracyComparison(prompt, groundTruth || undefined),
      {
        onSuccess: (data) => {
          setAccuracyComparison(data)
        },
        successMessage: '',
        errorMessage: 'Failed to run accuracy comparison',
        onFinally: () => setAccuracyLoading(false)
      }
    )
  }

  const handleSubmit = async () => {
    if (!prompt.trim()) return

    // Clear previous results and start loading
    setRouterResult(null)
    setBenchmarkResult(null)
    setAccuracyComparison(null)
    setRouterLoading(true)
    setBenchmarkLoading(true)
    setIsLoading(true)

    // Check if offline mode is enabled and we have replay data
    if (offlineMode && selectedScenario) {
      const modelRouterData = replayData.modelRouterScenarios as any
      const departmentData = modelRouterData[selectedDepartment]
      const scenarioData = departmentData?.[selectedScenario.id]

      if (scenarioData) {
        // Simulate router response with delay
        setTimeout(() => {
          setRouterResult({
            model_type: scenarioData.router.model,
            output: scenarioData.router.output,
            prompt_tokens: scenarioData.router.promptTokens,
            completion_tokens: scenarioData.router.completionTokens,
            total_tokens: scenarioData.router.totalTokens,
            server_processing_ms: scenarioData.router.serverProcessingMs,
            network_ms: scenarioData.router.networkMs,
            response_time_ms: scenarioData.router.responseTimeMs
          })
          setRouterLoading(false)
        }, scenarioData.router.responseTimeMs)

        // Simulate benchmark response with delay
        setTimeout(() => {
          setBenchmarkResult({
            model_type: scenarioData.benchmark.model,
            output: scenarioData.benchmark.output,
            prompt_tokens: scenarioData.benchmark.promptTokens,
            completion_tokens: scenarioData.benchmark.completionTokens,
            total_tokens: scenarioData.benchmark.totalTokens,
            server_processing_ms: scenarioData.benchmark.serverProcessingMs,
            network_ms: scenarioData.benchmark.networkMs,
            response_time_ms: scenarioData.benchmark.responseTimeMs
          })
          setBenchmarkLoading(false)
          setIsLoading(false)

          // Set accuracy comparison if available
          if (scenarioData.accuracy) {
            setTimeout(() => {
              setAccuracyComparison({
                scenario_id: selectedScenario.id,
                router: {
                  model_type: scenarioData.router.model,
                  output: scenarioData.router.output,
                  prompt_tokens: scenarioData.router.promptTokens,
                  completion_tokens: scenarioData.router.completionTokens,
                  total_tokens: scenarioData.router.totalTokens,
                  accuracy_evaluation: {
                    score: scenarioData.accuracy.router.score,
                    reasoning: scenarioData.accuracy.router.reasoning,
                    strengths: scenarioData.accuracy.router.strengths,
                    weaknesses: scenarioData.accuracy.router.weaknesses,
                    model_evaluated: scenarioData.router.model
                  }
                },
                benchmark: {
                  model_type: scenarioData.benchmark.model,
                  output: scenarioData.benchmark.output,
                  prompt_tokens: scenarioData.benchmark.promptTokens,
                  completion_tokens: scenarioData.benchmark.completionTokens,
                  total_tokens: scenarioData.benchmark.totalTokens,
                  accuracy_evaluation: {
                    score: scenarioData.accuracy.benchmark.score,
                    reasoning: scenarioData.accuracy.benchmark.reasoning,
                    strengths: scenarioData.accuracy.benchmark.strengths,
                    weaknesses: scenarioData.accuracy.benchmark.weaknesses,
                    model_evaluated: scenarioData.benchmark.model
                  }
                },
                timing: {
                  response_generation_ms: scenarioData.benchmark.responseTimeMs,
                  accuracy_evaluation_ms: 500,
                  total_ms: scenarioData.benchmark.responseTimeMs + 500
                }
              })
              setAccuracyLoading(false)
            }, scenarioData.benchmark.responseTimeMs + 500)
            setAccuracyLoading(true)
          }
        }, scenarioData.benchmark.responseTimeMs)

        return
      }
    }

    const BACKEND_URL = (import.meta as any).env?.VITE_BACKEND_URL || 'http://localhost:8000'

    // Create completely independent functions for each request
    const makeRouterRequest = () => {
      const startTime = performance.now()
      console.log('🚀 Router request started at:', startTime)
      
      return fetch(`${BACKEND_URL}/api/route`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          ...getAuthHeaders()
        },
        body: JSON.stringify({ prompt })
      })
      .then(response => {
        if (!response.ok) throw new Error(`Router API error: ${response.status}`)
        return response.json()
      })
      .then(data => {
        const endTime = performance.now()
        const totalTime = endTime - startTime
        const networkTime = Math.max(0, totalTime - (data.server_processing_ms || 0))
        
        console.log('✅ Router response received:', {
          startTime,
          endTime,
          totalTime,
          serverTime: data.server_processing_ms,
          networkTime
        })
        
        setRouterResult({
          ...data,
          response_time_ms: totalTime,
          network_ms: networkTime
        })
        setRouterLoading(false)
      })
      .catch(error => {
        console.error('❌ Router error:', error)
        setRouterLoading(false)
        toast({
          title: "Router Error",
          description: "Failed to get router response",
          variant: "destructive"
        })
      })
    }

    const makeBenchmarkRequest = () => {
      const startTime = performance.now()
      console.log('🚀 Benchmark request started at:', startTime)
      
      return fetch(`${BACKEND_URL}/api/benchmark`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          ...getAuthHeaders()
        },
        body: JSON.stringify({ prompt })
      })
      .then(response => {
        if (!response.ok) throw new Error(`Benchmark API error: ${response.status}`)
        return response.json()
      })
      .then(data => {
        const endTime = performance.now()
        const totalTime = endTime - startTime
        const networkTime = Math.max(0, totalTime - (data.server_processing_ms || 0))
        
        console.log('✅ Benchmark response received:', {
          startTime,
          endTime,
          totalTime,
          serverTime: data.server_processing_ms,
          networkTime
        })
        
        setBenchmarkResult({
          ...data,
          response_time_ms: totalTime,
          network_ms: networkTime
        })
        setBenchmarkLoading(false)
      })
      .catch(error => {
        console.error('❌ Benchmark error:', error)
        setBenchmarkLoading(false)
        toast({
          title: "Benchmark Error",
          description: "Failed to get benchmark response", 
          variant: "destructive"
        })
      })
    }

    // Execute both requests completely independently
    console.log('🎯 Starting independent API calls...')
    const routerPromise = makeRouterRequest()
    const benchmarkPromise = makeBenchmarkRequest()

    Promise.all([routerPromise, benchmarkPromise])
      .then(() => {
        setIsLoading(false)
      })
      .catch(() => {
        setIsLoading(false)
      })
  }

  const selectedDept = departments.find(d => d.value === selectedDepartment)
  const routerCost = routerResult ? calculateCost(
    routerResult.prompt_tokens,
    routerResult.completion_tokens,
    routerResult.model_type,
    pricing,
    true
  ) : null
  const benchmarkCost = benchmarkResult ? calculateCost(
    benchmarkResult.prompt_tokens,
    benchmarkResult.completion_tokens,
    benchmarkResult.model_type,
    pricing
  ) : null
  const savings = routerCost && benchmarkCost ? ((benchmarkCost.totalCost - routerCost.totalCost) / benchmarkCost.totalCost * 100) : null
  const fallbackModelsInView = Array.from(new Set(
    [
      routerCost?.usedFallback && routerResult ? routerResult.model_type : null,
      benchmarkCost?.usedFallback && benchmarkResult ? benchmarkResult.model_type : null
    ].filter((model): model is string => Boolean(model))
  ))
  const hasFallbackModels = fallbackModelsInView.length > 0
  const defaultPricing = pricing['default']
  const routerCostLabel = routerCost
    ? `$${routerCost.totalCost.toFixed(6)}${routerCost.classificationCost > 0 ? ` (includes $${routerCost.classificationCost.toFixed(6)} router classification)` : ''}`
    : '—'
  const benchmarkCostLabel = benchmarkCost
    ? `$${benchmarkCost.totalCost.toFixed(6)}`
    : '—'

  return (
    <div className="mx-auto max-w-6xl px-6 py-6 space-y-6">
      {/* Header */}
      <div className="text-center space-y-2">
        <div className="flex items-center justify-center gap-2 mb-2">
          <img src="/FoundryLogo.svg" alt="Foundry" className="h-8 w-8" />
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Foundry Model Router
          </h1>
        </div>
        <p className="text-slate-600 max-w-2xl mx-auto">
          Compare intelligent model routing against premium benchmarks. See real-time cost savings and performance metrics.
        </p>
      </div>

      {hasFallbackModels && (
        <Alert className="bg-amber-50 border-amber-200">
          <AlertDescription className="text-amber-800">
            Fallback pricing applied for: {fallbackModelsInView.join(', ')}. Azure likely routed to newer models that are not listed in <code className="px-1 bg-amber-100 rounded">pricing.json</code>. {defaultPricing ? `Fallback input/output: $${defaultPricing.input_per_1m.toFixed(2)} / $${defaultPricing.output_per_1m.toFixed(2)} per 1M tokens. ` : ''}The router classification surcharge ($${ROUTER_CLASSIFICATION_INPUT_RATE.toFixed(2)} per 1M input tokens) is added separately. Update the pricing file with the exact rates to remove this message.
          </AlertDescription>
        </Alert>
      )}



      <div className="space-y-6">
        {/* Scenario Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">Scenario</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              {/* Department Selection */}
              <div>
                <h4 className="font-medium text-sm mb-3 flex items-center gap-2">
                  {selectedDept && <selectedDept.icon className="h-4 w-4" />}
                  Department
                </h4>
                <Select value={selectedDepartment} onValueChange={setSelectedDepartment} disabled={offlineMode}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {departments.map((dept) => {
                      const Icon = dept.icon
                      const isDisabled = offlineMode && dept.value !== 'Finance'
                      return (
                        <SelectItem 
                          key={dept.value} 
                          value={dept.value}
                          disabled={isDisabled}
                        >
                          <div className="flex items-center gap-2">
                            <Icon className="h-4 w-4" />
                            {dept.label}
                          </div>
                        </SelectItem>
                      )
                    })}
                  </SelectContent>
                </Select>
              </div>

              {/* Scenarios */}
              <div>
                <h4 className="font-medium text-sm mb-3">Scenarios</h4>
                <p className="text-xs text-slate-600 mb-3">Select a pre-built scenario or create your own</p>
                <div className="space-y-2 max-h-120 overflow-y-auto">
                  {scenariosLoading ? (
                    <div className="space-y-2">
                      {[1, 2, 3].map(i => <Skeleton key={i} className="h-16 w-full" />)}
                    </div>
                  ) : (
                    scenarios.map((scenario) => (
                      <Card
                        key={scenario.id}
                        className={`cursor-pointer transition-all hover:shadow-md ${
                          selectedScenario?.id === scenario.id 
                            ? 'ring-2 ring-blue-500 bg-blue-50' 
                            : 'hover:bg-slate-50'
                        }`}
                        onClick={() => handleScenarioSelect(scenario)}
                      >
                        <CardContent className="p-3">
                          <div className="flex justify-between items-start mb-1">
                            <h4 className="font-medium text-sm">{scenario.title}</h4>
                            <Badge 
                              variant="secondary" 
                              className={`text-xs ${complexityColors[scenario.complexity]}`}
                            >
                              {scenario.complexity}
                            </Badge>
                          </div>
                          <p className="text-xs text-slate-600">{scenario.qualityExpectation}</p>
                        </CardContent>
                      </Card>
                    ))
                  )}
                </div>
              </div>

              {/* Prompt Input */}
              <div>
                <h4 className="font-medium text-sm mb-3">Prompt</h4>
                {selectedScenario && (
                  <div className="p-2 bg-blue-50 rounded text-xs mb-3">
                    <strong>Complexity:</strong> {selectedScenario.complexity} | 
                    <strong> Expected:</strong> {selectedScenario.qualityExpectation}
                  </div>
                )}
                <Textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Enter your prompt here or select a pre-built scenario..."
                  className="min-h-[120px] mb-3"
                  disabled={offlineMode}
                />
                
                {/* Ground Truth Collapsible */}
                <Collapsible className="mb-3">
                  <CollapsibleTrigger asChild>
                    <Button 
                      variant="outline" 
                      className="w-full justify-between"
                      disabled={offlineMode}
                    >
                      <span className="flex items-center gap-2">
                        <Target className="h-4 w-4" />
                        Ground Truth (Optional)
                      </span>
                      <ChevronDown className="h-4 w-4 transition-transform" />
                    </Button>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="mt-3 space-y-2">
                    <p className="text-xs text-slate-600">
                      If using a custom prompt, provide the expected answer or ground truth for accuracy comparison.
                    </p>
                    <Textarea
                      value={groundTruth}
                      onChange={(e) => setGroundTruth(e.target.value)}
                      placeholder="Enter the expected ground truth answer..."
                      className="min-h-[80px]"
                      disabled={offlineMode}
                    />
                  </CollapsibleContent>
                </Collapsible>

                <Button 
                  onClick={handleSubmit} 
                  disabled={!prompt.trim() || isLoading || (offlineMode && !selectedScenario)}
                  className="w-full"
                >
                  {isLoading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Zap className="h-4 w-4 mr-2" />
                      {selectedScenario ? 'Analyze with Both Models + Accuracy' : 'Analyze with Both Models'}
                    </>
                  )}
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Output Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">Output</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <ResponseCard
                title="Model Router"
                icon={Bot}
                isLoading={routerLoading}
                result={routerResult}
                pricing={pricing}
                includeRouterSurcharge
              />
              <ResponseCard
                title="Benchmark Model"
                icon={Cpu}
                isLoading={benchmarkLoading}
                result={benchmarkResult}
                pricing={pricing}
                isDark={true}
              />
            </div>
          </CardContent>
        </Card>

        {/* Comparison Section */}
        <Card>
          <CardHeader>
            <CardTitle className="text-xl">Comparison</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Cost Comparison */}
            <div>
              <h4 className="font-medium text-sm mb-3 flex items-center gap-2">
                <DollarSign className="h-4 w-4" />
                Cost Comparison
              </h4>
              {savings !== null ? (
                <Alert className="bg-green-50 border-green-200">
                  <DollarSign className="h-4 w-4 text-green-600" />
                  <AlertDescription className="text-green-800">
                    <strong>Cost Savings: {savings.toFixed(1)}%</strong> - 
                    Model Router: {routerCostLabel} vs Benchmark: {benchmarkCostLabel}
                  </AlertDescription>
                </Alert>
              ) : (
                <div className="p-4 bg-slate-50 border border-slate-200 rounded text-center text-sm text-slate-500">
                  <DollarSign className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>Submit a prompt to see cost comparison</p>
                </div>
              )}
            </div>

            {/* Latency Comparison */}
            <div>
              <h4 className="font-medium text-sm mb-3 flex items-center gap-2">
                <Clock className="h-4 w-4" />
                Latency Comparison
              </h4>
              {routerResult || benchmarkResult ? (
                <LatencyComparison 
                  routerResult={routerResult} 
                  benchmarkResult={benchmarkResult} 
                />
              ) : (
                <div className="p-4 bg-slate-50 border border-slate-200 rounded text-center text-sm text-slate-500">
                  <Clock className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>Submit a prompt to see latency comparison</p>
                </div>
              )}
            </div>

            {/* Accuracy Comparison */}
            <div>
              <h4 className="font-medium text-sm mb-3 flex items-center gap-2">
                <Target className="h-4 w-4" />
                Accuracy Comparison
              </h4>
              {(selectedScenario || groundTruth.trim().length > 0) && routerResult && benchmarkResult && !accuracyComparison ? (
                <div className="space-y-3">
                  <Alert className="bg-blue-50 border-blue-200">
                    <AlertDescription className="text-blue-800">
                      Both model responses are ready. Click the button below to evaluate accuracy.
                    </AlertDescription>
                  </Alert>
                  <Button 
                    onClick={handleRunAccuracyComparison}
                    disabled={accuracyLoading}
                    className="w-full"
                    variant="outline"
                  >
                    {accuracyLoading ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2" />
                        Evaluating Accuracy...
                      </>
                    ) : (
                      <>
                        <Target className="h-4 w-4 mr-2" />
                        Run Accuracy Comparison
                      </>
                    )}
                  </Button>
                </div>
              ) : (selectedScenario || groundTruth.trim().length > 0 || accuracyComparison) ? (
                accuracyLoading && !accuracyComparison ? (
                  <Card className="w-full">
                    <CardContent className="pt-6 space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-3">
                          <Skeleton className="h-8 w-full" />
                          <Skeleton className="h-4 w-full" />
                          <Skeleton className="h-2 w-full" />
                          <Skeleton className="h-20 w-full" />
                        </div>
                        <div className="space-y-3 md:border-l md:pl-4">
                          <Skeleton className="h-8 w-full" />
                          <Skeleton className="h-4 w-full" />
                          <Skeleton className="h-2 w-full" />
                          <Skeleton className="h-20 w-full" />
                        </div>
                      </div>
                      <div className="flex items-center justify-center gap-2 text-sm text-slate-600 pt-4">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-slate-600" />
                        Evaluating...
                      </div>
                    </CardContent>
                  </Card>
                ) : accuracyComparison ? (
                  <AccuracyComparison 
                    routerEvaluation={accuracyComparison.router.accuracy_evaluation}
                    benchmarkEvaluation={accuracyComparison.benchmark.accuracy_evaluation}
                    scenarioId={accuracyComparison.scenario_id}
                  />
                ) : (
                  <div className="p-4 bg-slate-50 border border-slate-200 rounded text-center text-sm text-slate-500">
                    <Target className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p>Submit a prompt to see model responses</p>
                  </div>
                )
              ) : (
                <div className="p-4 bg-slate-50 border border-slate-200 rounded text-center text-sm text-slate-500">
                  <Target className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>Select a scenario or provide ground truth to enable accuracy comparison</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}