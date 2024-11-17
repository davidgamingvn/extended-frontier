"use client";

import { AnimatedInstructions } from "@/components/ui/animated-instructions";
import { Button } from "@/components/ui/button";
import { FlipWords } from "@/components/ui/flip-words";
import { InfiniteMovingCards } from "@/components/ui/infinite-moving-cards";
import { ModeToggle } from "@/components/ui/mode-toggle";
import { TextGenerateEffect } from "@/components/ui/text-generate-effect";
import { features, steps } from "@/lib/utils";
import { Download, Wifi } from "lucide-react";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      <header className="px-4 lg:px-6 h-14 flex items-center">
        <Link className="flex items-center justify-center" href="#">
          <Wifi className="h-6 w-6 text-[#fe0036]" />
          <span className="sr-only">FrontierBeyond</span>
        </Link>
        <nav className="ml-auto flex items-center gap-4 sm:gap-6">
          <Link
            className="text-sm font-medium hover:underline underline-offset-4"
            href="#features"
          >
            Features
          </Link>
          <Link
            className="text-sm font-medium hover:underline underline-offset-4"
            href="#how-it-works"
          >
            How It Works
          </Link>
          <Link
            className="text-sm font-medium hover:underline underline-offset-4"
            href="#download"
          >
            Download
          </Link>
          <ModeToggle />
        </nav>
      </header>
      <main className="flex-1">
        <section className="w-full py-12 md:py-24 lg:py-32 xl:py-48">
          <>
            <div className="container px-4 md:px-6">
              <div className="flex flex-col items-center space-y-4 text-center">
                <div className="space-y-2">
                  <h1 className="text-3xl font-bold tracking-wide sm:text-4xl md:text-5xl lg:text-6xl/none">
                    Optimize Your WiFi with{" "}
                    <TextGenerateEffect words="FrontierBeyond" />
                  </h1>
                  <div className="mx-auto div-4 max-w-[700px] p-4 text-gray-500 md:text-xl dark:text-gray-400">
                    Extend your
                    <FlipWords words={["WiFi", "connectivity", "coverage"]} />
                  </div>
                </div>
                <div className="space-x-4">
                  <Link href="/home">
                    <Button variant="ghost" className="bg-[#fe0036] text-white">
                      Get started
                    </Button>
                  </Link>
                  <Link href="#how-it-works">
                    <Button variant="outline">Learn More</Button>
                  </Link>
                </div>
              </div>
            </div>
          </>
        </section>
        <section id="features" className="w-full">
          <div className="container px-4 md:px-6">
            <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl text-center mb-12">
              Key Features
            </h2>
            <InfiniteMovingCards
              className="mx-auto"
              speed="slow"
              pauseOnHover={false}
              items={features}
            />
          </div>
        </section>
        <section id="how-it-works" className="w-full py-12 md:py-24 lg:py-32">
          <div className="container px-4 md:px-6">
            <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl text-center mb-12">
              How It Works
            </h2>
            <AnimatedInstructions steps={steps} autoplay={true} />
          </div>
        </section>
        <section
          id="download"
          className="w-full py-12 md:py-24 lg:py-32 bg-[#fe0036] text-white"
        >
          <div className="container px-4 md:px-6">
            <div className="flex flex-col items-center space-y-6 text-center">
              <h2 className="text-3xl font-bold tracking-tighter sm:text-5xl">
                Ready to Boost Your WiFi?
              </h2>
              <p className="mx-auto max-w-[600px] text-white/90 md:text-xl">
                Download the FrontierBeyond app now and experience seamless
                connectivity throughout your home.
              </p>
              <div className="flex items-center p-2 space-x-4">
                <Button className="bg-white text-[#fe0036] hover:bg-gray-100">
                  <Download className="mr-2 h-4 w-4" /> Download now
                </Button>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
