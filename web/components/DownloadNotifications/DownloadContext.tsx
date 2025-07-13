'use client';

import React, { createContext, useContext, useEffect, useState, useRef, useCallback } from 'react';

interface DownloadProgress {
  id: number;
  name: string;
  progress: number;
  status: 'pending' | 'downloading' | 'completed' | 'failed';
  message?: string | any; // Allow any to handle unexpected API responses
}

interface DownloadContextType {
  downloads: DownloadProgress[];
  addDownload: (download: Omit<DownloadProgress, 'progress'> & { progress?: number }) => void;
  updateDownload: (id: number, data: Partial<DownloadProgress>) => void;
  removeDownload: (id: number) => void;
  clearCompleted: () => void;
}

// Helper to safely process message from API
const safeMessageHandler = (message: any): string => {
  if (message === null || message === undefined) {
    return '';
  }
  if (typeof message === 'string') {
    return message;
  }
  if (typeof message === 'object') {
    try {
      return JSON.stringify(message);
    } catch (error) {
      console.error('Error stringifying message object:', error);
      return 'Error processing message';
    }
  }
  return String(message);
};

const DownloadContext = createContext<DownloadContextType | undefined>(undefined);

export const useDownloads = () => {
  const context = useContext(DownloadContext);
  if (context === undefined) {
    throw new Error('useDownloads must be used within a DownloadProvider');
  }
  return context;
};

export const DownloadProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [downloads, setDownloads] = useState<DownloadProgress[]>([]);
  // Use a ref to access latest state without dependencies
  const downloadsRef = useRef<DownloadProgress[]>([]);

  // Update ref when state changes
  useEffect(() => {
    downloadsRef.current = downloads;
  }, [downloads]);

  // Load saved downloads from localStorage on initialization
  useEffect(() => {
    const savedDownloads = localStorage.getItem('activeDownloads');
    if (savedDownloads) {
      try {
        setDownloads(JSON.parse(savedDownloads));
      } catch (e) {
        console.error('Failed to parse saved downloads:', e);
        localStorage.removeItem('activeDownloads');
      }
    }
  }, []);

  // Save downloads to localStorage when they change
  useEffect(() => {
    if (downloads.length > 0) {
      localStorage.setItem('activeDownloads', JSON.stringify(downloads));
    } else {
      localStorage.removeItem('activeDownloads');
    }
  }, [downloads]);

  // Poll for status updates for active downloads
  useEffect(() => {
    // Create a polling function that can be called repeatedly
    const pollDownloads = async () => {
      const currentDownloads = downloadsRef.current;
      const activeDownloads = currentDownloads.filter(
        (d) => d.status === 'pending' || d.status === 'downloading'
      );

      if (activeDownloads.length === 0) return;

      const updatedDownloads = [...currentDownloads];
      let hasChanges = false;

      for (const download of activeDownloads) {
        try {
          const response = await fetch(
            `${process.env.NEXT_PUBLIC_API_URL}/datasets/${download.id}/status?status_type=download`
          );

          if (response.ok) {
            const data = await response.json();
            const index = updatedDownloads.findIndex((d) => d.id === download.id);

            if (index !== -1) {
              const newStatus =
                data.status === 'completed'
                  ? 'completed'
                  : data.status === 'failed'
                    ? 'failed'
                    : 'downloading';

              const newProgress = data.progress !== undefined ? data.progress : download.progress;

              // Safely process the message to ensure it's a string
              const processedMessage =
                data.message !== undefined ? safeMessageHandler(data.message) : download.message;

              if (
                newStatus !== download.status ||
                newProgress !== download.progress ||
                processedMessage !== download.message
              ) {
                updatedDownloads[index] = {
                  ...updatedDownloads[index],
                  status: newStatus,
                  progress: newProgress,
                  message: processedMessage,
                };
                hasChanges = true;
              }
            }
          }
        } catch (error) {
          console.error(`Error checking status for download ${download.id}:`, error);
        }
      }

      if (hasChanges) {
        setDownloads(updatedDownloads);
      }
    };

    // Set up polling interval
    const intervalId = setInterval(pollDownloads, 3000);

    // Clear interval on cleanup
    return () => clearInterval(intervalId);
  }, []); // Empty dependency array since we use refs

  const addDownload = useCallback(
    (download: Omit<DownloadProgress, 'progress'> & { progress?: number }) => {
      setDownloads((prev) => {
        // Check if download already exists
        const exists = prev.some((d) => d.id === download.id);
        if (exists) {
          return prev.map((d) => {
            if (d.id === download.id) {
              return {
                ...d,
                status: download.status,
                message:
                  typeof download.message === 'string'
                    ? download.message
                    : safeMessageHandler(download.message),
              };
            }
            return d;
          });
        }

        // Add new download with default progress of 0 and process message
        return [
          ...prev,
          {
            ...download,
            progress: download.progress || 0,
            message:
              typeof download.message === 'string'
                ? download.message
                : safeMessageHandler(download.message),
          },
        ];
      });
    },
    []
  );

  const updateDownload = useCallback((id: number, data: Partial<DownloadProgress>) => {
    setDownloads((prev) =>
      prev.map((download) => {
        if (download.id === id) {
          // Process message if it's being updated
          const updatedMessage =
            data.message !== undefined
              ? typeof data.message === 'string'
                ? data.message
                : safeMessageHandler(data.message)
              : download.message;

          return {
            ...download,
            ...data,
            message: updatedMessage,
          };
        }
        return download;
      })
    );
  }, []);

  const removeDownload = useCallback((id: number) => {
    setDownloads((prev) => prev.filter((download) => download.id !== id));
  }, []);

  const clearCompleted = useCallback(() => {
    setDownloads((prev) =>
      prev.filter((download) => download.status !== 'completed' && download.status !== 'failed')
    );
  }, []);

  const value = {
    downloads,
    addDownload,
    updateDownload,
    removeDownload,
    clearCompleted,
  };

  return <DownloadContext.Provider value={value}>{children}</DownloadContext.Provider>;
};
