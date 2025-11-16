import React, { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, CheckCircle } from 'lucide-react'
import { Button } from "@/components/ui/button"
import { parseStoredAuthToken } from "@/utils/auth"

export default function HomePage() {
  const [userId, setUserId] = useState<string | null>(null)

  useEffect(() => {
    const stored = parseStoredAuthToken()
    if (stored) {
      setUserId(stored.id)
    }
  }, [])

  return (
    <div className="flex flex-col min-h-screen">
      <main className="flex-grow">
        <section className="w-full py-8 md:py-16 lg:py-20 xl:py-24 hero">
          <div className="flex flex-col items-center space-y-4 text-center">
            <div className="space-y-2">
              <h1 className="text-4xl font-bold tracking-tighter sm:text-4xl md:text-5xl lg:text-6xl/none text-gray-900">
                Microsoft Foundry Model Router
              </h1>
              <p className="mx-auto max-w-[700px] text-gray-600 md:text-xl">
                Intelligently select the best LLM for every prompt. Save costs while maintaining quality by routing requests to the most suitable model.
              </p>
              {userId && (
                <p className="mx-auto max-w-[700px] text-gray-600 md:text-xl">
                  Logged in as <span className="font-semibold">{userId}</span>
                </p>
              )}
            </div>
          </div>
        </section>
        <section className="w-full py-4 md:py-6 lg:py-8 bg-white">
          <div className="flex flex-col items-center justify-center space-y-4 text-center">
            <div className="space-x-4">
              <Button asChild size="lg">
                <Link to="/zava-sportswear">Zava eCommerce Demo</Link>
              </Button>
              <Button asChild variant="outline" size="lg">
                <Link to="/model-router">View Performance Metrics</Link>
              </Button>
            </div>
          </div>
        </section>
        <section className="w-full py-8 md:py-16 lg:py-24 bg-background">
          <div className="container px-4 md:px-6">
            <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl text-center mb-12">Why Use Model Router?</h2>
            <div className="grid gap-10 sm:grid-cols-2 md:grid-cols-3">
              <div className="flex flex-col items-center text-center">
                <CheckCircle className="h-12 w-12 text-primary mb-4" />
                <h3 className="text-xl font-bold mb-2">Intelligent Routing</h3>
                <p className="text-muted-foreground">Automatically selects the best LLM based on query complexity and requirements.</p>
              </div>
              <div className="flex flex-col items-center text-center">
                <CheckCircle className="h-12 w-12 text-primary mb-4" />
                <h3 className="text-xl font-bold mb-2">Cost Optimization</h3>
                <p className="text-muted-foreground">Uses smaller, cheaper models when sufficient, while maintaining quality for complex tasks.</p>
              </div>
              <div className="flex flex-col items-center text-center">
                <CheckCircle className="h-12 w-12 text-primary mb-4" />
                <h3 className="text-xl font-bold mb-2">Single Deployment</h3>
                <p className="text-muted-foreground">Deploy once and access multiple LLMs through a unified chat interface.</p>
              </div>
            </div>
          </div>
        </section>
        <section className="w-full py-8 md:py-16 lg:py-20 bg-white">
          <div className="container px-4 md:px-6">
            <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl text-center mb-12">Video Walkthrough</h2>
            <div className="max-w-4xl mx-auto">
              <div style={{padding: '52.5% 0 0 0', position: 'relative'}}>
                <iframe 
                  src="https://player.vimeo.com/video/1128712354?title=0&amp;byline=0&amp;portrait=0&amp;badge=0&amp;autopause=0&amp;player_id=0&amp;app_id=58479&amp;muted=0" 
                  frameBorder="0" 
                  allow="autoplay; fullscreen; picture-in-picture; clipboard-write; encrypted-media; web-share" 
                  referrerPolicy="strict-origin-when-cross-origin" 
                  style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%'}} 
                  title="Foundry Model Router"
                />
              </div>
            </div>
          </div>
        </section>
        <section className="w-full py-12 md:py-24 lg:py-32 bg-muted">
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center justify-center space-y-4 text-center">
              <div className="space-y-2">
                <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">Explore Model Router</h2>
                <p className="mx-auto max-w-[600px] text-muted-foreground md:text-xl">
                  See how intelligent LLM routing delivers high performance and cost savings in real-world applications.
                </p>
              </div>
              <div className="space-x-4">
                <Button asChild size="lg">
                  <Link to="/zava-sportswear">
                    E-Commerce Demo
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
                <Button asChild variant="secondary" size="lg">
                  <Link to="/model-router">
                    Performance Metrics
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}