  import { useState, useEffect } from 'react'
  import { Hero } from '@/components/zava/Hero'
  import { Products } from '@/components/zava/Products'
  import { Technology } from '@/components/zava/Technology'
  import { Athletes } from '@/components/zava/Athletes'
  import { About } from '@/components/zava/About'
  import { Contact } from '@/components/zava/Contact'
  import { ZavaChatbot } from '@/components/zava/ZavaChatbot'
  import { Lightning, X } from '@phosphor-icons/react'
  import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog'
  import { Button } from '@/components/ui/button'

  export default function ZavaSportswearPage() {
    const [activeSection, setActiveSection] = useState('home')
    const [showDemoModal, setShowDemoModal] = useState(true)

    const scrollToSection = (sectionId: string) => {
      setActiveSection(sectionId)
      const element = document.getElementById(sectionId)
      if (element) {
        element.scrollIntoView({ behavior: 'smooth' })
      }
    }

    return (
      <div className="min-h-screen bg-background zava-page">
        {/* Demo Instructions Modal */}
        <Dialog open={showDemoModal} onOpenChange={setShowDemoModal}>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2">
                <Lightning className="w-5 h-5 text-primary" weight="bold" />
                Welcome to Zava Demo
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <DialogDescription className="text-base text-foreground">
                This is a demo of a fictitious SportsTech company, <span className="font-semibold">Zava</span>.
              </DialogDescription>
              
              <div className="space-y-3 text-sm text-muted-foreground">
                <p>
                  You can toggle the chatbot to use the <span className="text-foreground font-medium">Foundry Model Router</span>, or a <span className="text-foreground font-medium">benchmark model (GPT-5)</span>.
                </p>
                
                <p>
                  Observe how the model-router <span className="text-foreground font-medium">dynamically chooses the best model</span>. For simple questions, this will be <span className="text-foreground font-medium">much faster</span>, delivering a better customer experience!
                </p>
              </div>

              <div className="pt-4 flex gap-2">
                <Button 
                  onClick={() => setShowDemoModal(false)}
                  className="flex-1"
                >
                  Get Started
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* Internal Navigation Bar for Zava */}
        <div className="sticky top-0 z-40 bg-background/95 backdrop-blur-sm border-b">
          <div className="container mx-auto px-4 h-14 flex items-center justify-between">
            {/* Zava Logo */}
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <Lightning className="w-5 h-5 text-primary-foreground" weight="bold" />
              </div>
              <span className="text-xl font-bold text-primary">ZAVA</span>
            </div>

            {/* Navigation Links */}
            <div className="hidden md:flex items-center space-x-6">
              {[
                { name: 'Home', id: 'home' },
                { name: 'Products', id: 'products' },
                { name: 'Technology', id: 'technology' },
                { name: 'Athletes', id: 'athletes' },
                { name: 'About', id: 'about' },
                { name: 'Contact', id: 'contact' },
              ].map((item) => (
                <button
                  key={item.id}
                  onClick={() => scrollToSection(item.id)}
                  className={`text-sm font-medium transition-colors hover:text-accent ${
                    activeSection === item.id ? 'text-accent' : 'text-foreground'
                  }`}
                >
                  {item.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <main>
          <Hero id="home" onExploreClick={() => scrollToSection('products')} />
          <Products id="products" />
          <Technology id="technology" />
          <Athletes id="athletes" />
          <About id="about" />
          <Contact id="contact" />
        </main>

        {/* Footer */}
        <footer className="bg-primary text-primary-foreground py-8 md:py-12">
          <div className="container mx-auto px-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {/* Brand */}
              <div>
                <div className="flex items-center space-x-2 mb-4">
                  <div className="w-8 h-8 bg-primary-foreground rounded-lg flex items-center justify-center">
                    <Lightning className="w-5 h-5 text-primary" weight="bold" />
                  </div>
                  <span className="text-xl font-bold">ZAVA</span>
                </div>
                <p className="text-sm text-primary-foreground/80">
                  Revolutionizing athletic performance with cutting-edge smart sportswear technology.
                </p>
              </div>

              {/* Quick Links */}
              <div>
                <h4 className="font-semibold mb-4">Quick Links</h4>
                <ul className="space-y-2 text-sm">
                  <li>
                    <button 
                      onClick={() => scrollToSection('products')} 
                      className="text-primary-foreground/80 hover:text-primary-foreground transition-colors"
                    >
                      Products
                    </button>
                  </li>
                  <li>
                    <button 
                      onClick={() => scrollToSection('technology')} 
                      className="text-primary-foreground/80 hover:text-primary-foreground transition-colors"
                    >
                      Technology
                    </button>
                  </li>
                  <li>
                    <button 
                      onClick={() => scrollToSection('athletes')} 
                      className="text-primary-foreground/80 hover:text-primary-foreground transition-colors"
                    >
                      Athletes
                    </button>
                  </li>
                  <li>
                    <button 
                      onClick={() => scrollToSection('about')} 
                      className="text-primary-foreground/80 hover:text-primary-foreground transition-colors"
                    >
                      About
                    </button>
                  </li>
                </ul>
              </div>

              {/* Contact */}
              <div>
                <h4 className="font-semibold mb-4">Contact</h4>
                <ul className="space-y-2 text-sm text-primary-foreground/80">
                  <li>Email: info@zava.com</li>
                  <li>Phone: +1 (555) 123-4567</li>
                  <li>Address: 123 Tech Street, Innovation City</li>
                </ul>
              </div>
            </div>

            <div className="border-t border-primary-foreground/20 mt-8 pt-8 text-center text-sm text-primary-foreground/60">
              <p>&copy; 2025 Zava Smart Sportswear. All rights reserved.</p>
              <p className="mt-2">
                Design provided by{' '}
                <a 
                  href="https://github.com/patrick-vuong/zava-smart-sportswear/tree/main" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="underline hover:text-primary-foreground/80 transition-colors"
                >
                  patrick-vuong/zava-smart-sportswear
                </a>
              </p>
            </div>
          </div>
        </footer>

        {/* Chatbot */}
        <ZavaChatbot defaultOpen={true} />
      </div>
    )
  }
