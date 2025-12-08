// Navbar Component

'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Car, MessageSquare, Settings, Activity, Menu } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useTelemetryStore } from '@/stores/telemetryStore';
import { cn } from '@/lib/utils';

interface NavbarProps {
    onMenuClick?: () => void;
}

export function Navbar({ onMenuClick }: NavbarProps) {
    const pathname = usePathname();
    const isConnected = useTelemetryStore((state) => state.isConnected);
    const notifications = useTelemetryStore((state) => state.notifications);
    const unreadCount = notifications.filter(n => !n.read).length;

    const navItems = [
        { href: '/', label: 'Dashboard', icon: Activity },
        { href: '/vehicles', label: 'Vehicles', icon: Car },
        { href: '/admin', label: 'Admin', icon: Settings },
    ];

    return (
        <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container flex h-16 items-center">
                {/* Mobile menu button */}
                <Button
                    variant="ghost"
                    size="icon"
                    className="mr-2 md:hidden"
                    onClick={onMenuClick}
                >
                    <Menu className="h-5 w-5" />
                </Button>

                {/* Logo */}
                <Link href="/" className="flex items-center space-x-2 mr-6">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary">
                        <Car className="h-5 w-5 text-primary-foreground" />
                    </div>
                    <span className="font-bold text-xl hidden sm:inline-block">
                        SentinEV
                    </span>
                </Link>

                {/* Navigation */}
                <nav className="hidden md:flex items-center space-x-6 text-sm font-medium">
                    {navItems.map((item) => {
                        const Icon = item.icon;
                        const isActive = pathname === item.href ||
                            (item.href !== '/' && pathname.startsWith(item.href));

                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={cn(
                                    'flex items-center space-x-2 transition-colors hover:text-foreground/80',
                                    isActive ? 'text-foreground' : 'text-foreground/60'
                                )}
                            >
                                <Icon className="h-4 w-4" />
                                <span>{item.label}</span>
                            </Link>
                        );
                    })}
                </nav>

                {/* Right side */}
                <div className="ml-auto flex items-center space-x-4">
                    {/* Connection status */}
                    <div className="flex items-center space-x-2">
                        <div
                            className={cn(
                                'h-2 w-2 rounded-full',
                                isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                            )}
                        />
                        <span className="text-xs text-muted-foreground hidden sm:inline">
                            {isConnected ? 'Live' : 'Offline'}
                        </span>
                    </div>

                    {/* Notifications */}
                    {unreadCount > 0 && (
                        <Badge variant="destructive" className="text-xs">
                            {unreadCount}
                        </Badge>
                    )}
                </div>
            </div>
        </header>
    );
}
