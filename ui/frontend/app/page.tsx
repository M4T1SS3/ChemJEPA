'use client'

import { useState } from 'react'
import dynamic from 'next/dynamic'
import Card from '@/components/ui/Card'
import Button from '@/components/ui/Button'
import PropertiesTable from '@/components/tables/PropertiesTable'
import EnergyBar from '@/components/charts/EnergyBar'

// Dynamically import viewers to avoid SSR issues
const MoleculeViewer2D = dynamic(
  () => import('@/components/viewers/MoleculeViewer2D'),
  { ssr: false }
)

const MoleculeViewer3D = dynamic(
  () => import('@/components/viewers/MoleculeViewer3D'),
  { ssr: false }
)

export default function Home() {
  const [smiles, setSmiles] = useState('')
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d')
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState<any>(null)

  const handleAnalyze = async () => {
    if (!smiles) return

    setAnalyzing(true)

    try {
      // Call backend API
      const response = await fetch('http://localhost:8001/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ smiles }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Analysis failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (error: any) {
      console.error('Analysis error:', error)
      alert(`Error: ${error.message}`)
    } finally {
      setAnalyzing(false)
    }
  }

  const exampleMolecules = [
    { name: 'Ethanol', smiles: 'CCO' },
    { name: 'Benzene', smiles: 'c1ccccc1' },
    { name: 'Aspirin', smiles: 'CC(=O)Oc1ccccc1C(=O)O' },
    { name: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
  ]

  return (
    <main className="min-h-screen bg-gradient-to-br from-primary-50 via-background to-primary-50/30">
      {/* Fixed Header with Glass Effect */}
      <header className="fixed top-0 left-0 right-0 z-50 glass border-b border-border-light">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold gradient-text">ChemJEPA</h1>
              <p className="text-sm text-text-secondary mt-0.5">
                Hierarchical Latent World Models for Molecular Discovery
              </p>
            </div>
            <div className="flex gap-3">
              <Button variant="ghost" size="sm">
                Docs
              </Button>
              <Button variant="outline" size="sm">
                Export Results
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content - with top padding for fixed header */}
      <div className="pt-24 pb-16">
        <div className="max-w-7xl mx-auto px-6">
          {/* Hero Section - Molecule Input */}
          <section className="mb-12 animate-fade-in">
            <div className="text-center mb-8">
              <h2 className="text-4xl font-bold text-text mb-3">
                Analyze Molecular Structures
              </h2>
              <p className="text-lg text-text-secondary max-w-2xl mx-auto">
                Enter a SMILES string to explore molecular properties, visualize structures,
                and discover similar compounds
              </p>
            </div>

            <Card hover={false} className="max-w-4xl mx-auto">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-text-secondary mb-3">
                    SMILES String
                  </label>
                  <div className="flex gap-3">
                    <input
                      type="text"
                      value={smiles}
                      onChange={(e) => setSmiles(e.target.value)}
                      onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                      placeholder="e.g., CCO (ethanol)"
                      className="flex-1 px-4 py-3 bg-surface border border-border rounded-xl input-focus text-text placeholder:text-text-muted text-lg"
                    />
                    <Button onClick={handleAnalyze} isLoading={analyzing} size="lg">
                      {analyzing ? 'Analyzing...' : 'Analyze'}
                    </Button>
                  </div>
                </div>

                {/* Quick Examples */}
                <div className="flex flex-wrap gap-2">
                  <span className="text-sm text-text-muted">Try:</span>
                  {exampleMolecules.map((mol) => (
                    <button
                      key={mol.smiles}
                      onClick={() => setSmiles(mol.smiles)}
                      className="px-3 py-1.5 text-sm bg-primary-50 text-primary-600 rounded-lg hover:bg-primary-100 transition-colors"
                    >
                      {mol.name}
                    </button>
                  ))}
                </div>
              </div>
            </Card>
          </section>

          {/* Results Section */}
          {result && (
            <div className="space-y-8 animate-slide-up">
              {/* Large Molecular Viewer */}
              <section>
                <Card hover={false} className="overflow-hidden">
                  <div className="flex justify-between items-center mb-6">
                    <h3 className="text-2xl font-semibold text-text">Molecular Structure</h3>
                    {/* 2D/3D Toggle */}
                    <div className="flex gap-2 bg-surface p-1 rounded-xl">
                      <Button
                        size="sm"
                        variant={viewMode === '2d' ? 'primary' : 'ghost'}
                        onClick={() => setViewMode('2d')}
                      >
                        2D View
                      </Button>
                      <Button
                        size="sm"
                        variant={viewMode === '3d' ? 'primary' : 'ghost'}
                        onClick={() => setViewMode('3d')}
                      >
                        3D View
                      </Button>
                    </div>
                  </div>

                  <div className="flex justify-center bg-gradient-to-br from-surface to-primary-50/20 rounded-xl p-8">
                    {viewMode === '2d' ? (
                      <MoleculeViewer2D smiles={result.properties.smiles} width={800} height={600} />
                    ) : (
                      <MoleculeViewer3D smiles={result.properties.smiles} width={800} height={600} />
                    )}
                  </div>
                </Card>
              </section>

              {/* Properties Grid */}
              <section>
                <div className="grid md:grid-cols-2 gap-6">
                  {/* Molecular Properties */}
                  <Card hover={false}>
                    <h3 className="text-xl font-semibold text-text mb-6">Molecular Properties</h3>
                    <PropertiesTable properties={result.properties} />
                  </Card>

                  {/* Energy Decomposition */}
                  <Card hover={false}>
                    <h3 className="text-xl font-semibold text-text mb-6">Energy Decomposition</h3>
                    <EnergyBar energy={result.energy} />
                  </Card>
                </div>
              </section>

              {/* Discover Similar - Placeholder */}
              <section>
                <Card hover={false}>
                  <h3 className="text-xl font-semibold text-text mb-4">Similar Molecules</h3>
                  <p className="text-text-muted text-center py-8">
                    Similar molecule discovery coming soon...
                  </p>
                </Card>
              </section>
            </div>
          )}

          {/* Empty State */}
          {!result && !analyzing && (
            <div className="text-center py-16 animate-fade-in">
              <div className="text-6xl mb-4">🧪</div>
              <h3 className="text-2xl font-semibold text-text mb-2">Ready to Analyze</h3>
              <p className="text-text-muted max-w-md mx-auto">
                Enter a SMILES string above to get started with molecular analysis and visualization
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
