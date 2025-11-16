import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link, useLocation } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import { ChevronRight, Settings } from "lucide-react";

import { Breadcrumb, BreadcrumbItem, BreadcrumbLink, BreadcrumbList, BreadcrumbPage, BreadcrumbSeparator } from "@/components/ui/breadcrumb";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Separator } from "@/components/ui/separator";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarProvider,
  SidebarRail,
  SidebarTrigger,
} from "@/components/ui/sidebar";

import HomePage from '../pages/HomePage';
import ModelRouterPage from '../pages/ModelRouterPage';
import ZavaSportswearPage from '../pages/ZavaSportswearPage';
import DatasetEvaluationPage from '../pages/DatasetEvaluationPage';
import SettingsPage from '../pages/SettingsPage';

const CustomSidebarContent = () => {
  const location = useLocation();

  const navigationData = {
    navMain: [
      {
        title: "AI Demos",
        url: "#",
        icon: () => <img src="/FoundryLogo.svg" alt="Foundry" className="h-4 w-4" />,
        items: [
          {
            title: "Model Router Comparison",
            url: "/model-router",
          },
          {
            title: "Zava Sportswear Demo",
            url: "/zava-sportswear",
          },
          {
            title: "Dataset Evaluation",
            url: "/dataset-evaluation",
          }
        ],
      },
      {
        title: "Settings",
        url: "/settings",
        icon: Settings,
        items: [],
      },
    ]
  };

  return (
    <>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <Link to="/">
              <SidebarMenuButton size="lg">
                <div className="flex aspect-square size-8 items-center justify-center">
                  <img 
                    src="/FoundryLogo.svg" 
                    alt="Foundry Logo"
                    className="h-full w-full object-contain"
                  />
                </div>
                <span className="font-semibold">Model Router</span>
              </SidebarMenuButton>
            </Link>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Platform</SidebarGroupLabel>
          <SidebarMenu>
            {navigationData.navMain.map((item) => (
              <Collapsible
                key={item.title}
                asChild
                defaultOpen={item.title === "AI Demos"}
                className="group/collapsible"
              >
                <SidebarMenuItem>
                  {item.items.length > 0 ? (
                    <>
                      <CollapsibleTrigger asChild>
                        <SidebarMenuButton tooltip={item.title}>
                          {item.icon && <item.icon />}
                          <span>{item.title}</span>
                          <ChevronRight className="ml-auto transition-transform duration-200 group-data-[state=open]/collapsible:rotate-90" />
                        </SidebarMenuButton>
                      </CollapsibleTrigger>
                      <CollapsibleContent>
                        <SidebarMenuSub>
                          {item.items.map((subItem) => (
                            <SidebarMenuSubItem key={subItem.title}>
                              <SidebarMenuSubButton asChild>
                                <Link to={subItem.url}>
                                  <span>{subItem.title}</span>
                                </Link>
                              </SidebarMenuSubButton>
                            </SidebarMenuSubItem>
                          ))}
                        </SidebarMenuSub>
                      </CollapsibleContent>
                    </>
                  ) : (
                    <SidebarMenuButton asChild tooltip={item.title}>
                      <Link to={item.url}>
                        {item.icon && <item.icon />}
                        <span>{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  )}
                </SidebarMenuItem>
              </Collapsible>
            ))}
          </SidebarMenu>
        </SidebarGroup>
      </SidebarContent>
      <SidebarRail />
      
      {/* Footer with attribution and logo */}
      <div className="mt-auto p-4 flex flex-col items-start gap-1 text-xs font-semibold">
        <div className="flex items-center gap-2 text-black">
          <img src="/gbb-logo.svg" alt="GBB Logo" className="h-5 w-5" />
          <span>AI GBB</span>
        </div>
        <span className="text-gray-400 pl-7">Luca Stamatescu</span>
        <div className="flex items-center gap-2 text-black mt-2">
          <img src="/FoundryLogo.svg" alt="Foundry Logo" className="h-5 w-5" />
          <span>Product Group</span>
        </div>
        <span className="text-gray-400 pl-7">Sanjeev Jagtap</span>
      </div>
    </>
  );
};

const BreadcrumbHeader = () => {
  const location = useLocation();
  const pathnames = location.pathname.split('/').filter((x) => x);

  return (
    <header className="flex h-16 shrink-0 items-center gap-2 transition-[width,height] ease-linear group-has-[[data-collapsible=icon]]/sidebar-wrapper:h-12">
      <div className="flex items-center gap-2 px-4">
        <SidebarTrigger className="-ml-1" />
        <Separator orientation="vertical" className="mr-2 h-4" />
        <Breadcrumb>
          <BreadcrumbList>
            <BreadcrumbItem className="hidden md:block">
              <BreadcrumbLink asChild>
                <Link to="/">Home</Link>
              </BreadcrumbLink>
            </BreadcrumbItem>
            {pathnames.map((name, index) => {
              const routeTo = `/${pathnames.slice(0, index + 1).join('/')}`;
              const isLast = index === pathnames.length - 1;

              return (
                <React.Fragment key={name}>
                  <BreadcrumbSeparator className="hidden md:block" />
                  <BreadcrumbItem>
                    {isLast ? (
                      <BreadcrumbPage>{name.charAt(0).toUpperCase() + name.slice(1)}</BreadcrumbPage>
                    ) : (
                      <BreadcrumbLink asChild>
                        <Link to={routeTo}>
                          {name.charAt(0).toUpperCase() + name.slice(1)}
                        </Link>
                      </BreadcrumbLink>
                    )}
                  </BreadcrumbItem>
                </React.Fragment>
              );
            })}
          </BreadcrumbList>
        </Breadcrumb>
      </div>
    </header>
  );
};

interface LayoutProps {
  isDevelopment?: boolean
}

const LayoutContent = () => {
  const { ready, authorized } = useAuth();

  if (!ready) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-muted-foreground">Authenticating...</p>
      </div>
    );
  }

  if (!authorized) {
    return null;
  }

  return (
    <SidebarProvider>
      <div className="flex min-h-screen">
        <Sidebar collapsible="icon">
          <CustomSidebarContent />
        </Sidebar>
        <SidebarInset>
          <BreadcrumbHeader />
          <main className="flex-1">
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/model-router" element={<ModelRouterPage />} />
              <Route path="/dataset-evaluation" element={<DatasetEvaluationPage />} />
              <Route path="/zava-sportswear" element={<ZavaSportswearPage />} />
              <Route path="/settings" element={<SettingsPage />} />
            </Routes>
          </main>
        </SidebarInset>
      </div>
    </SidebarProvider>
  );
};

export default function Layout(_: LayoutProps) {
  return (
    <Router>
      <LayoutContent />
    </Router>
  );
}
