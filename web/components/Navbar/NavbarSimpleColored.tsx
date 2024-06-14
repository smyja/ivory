"use client";
import { useState,useEffect } from "react";
import { Loader } from "@mantine/core";
import {
  IconBellRinging,
  IconFingerprint,
  IconKey,
  IconSettings,
  Icon2fa,
 
  IconReceipt2,
  IconSwitchHorizontal,
  IconLogout,
  IconMessageChatbot,
  IconFiles
} from "@tabler/icons-react";

import Link  from 'next/link'
import { usePathname,useRouter } from 'next/navigation'
import { NavigationProgress,nprogress } from '@mantine/nprogress';
import classes from "./NavbarSimpleColored.module.css";

const data = [
  { link: "/dashboard", label: "Home", icon: IconBellRinging },
  { link: "/dashboard/chat", label: "Chat", icon: IconMessageChatbot },
  { link: "/dashboard/connectors", label: "Connectors", icon: Icon2fa },
  { link: "/dashboard/documents", label: "Documents", icon: IconFiles },
  { link: "/dashboard/datasets", label: "Datasets", icon: IconFiles },
  { link: "/dashboard/insight", label: "Insight", icon: IconFingerprint },
  { link: "/dashboard/settings", label: "Settings", icon: IconSettings },
];

export function NavbarSimpleColored() {
 const pathname = usePathname()
 const findLabelByPath = (path) => {
  const matchingItem = data.find(item => item.link === path);
  return matchingItem ? matchingItem.label : "Home"; 
};

// Set the initial active label based on the current pathname
const [active, setActive] = useState(findLabelByPath(pathname));

 useEffect(() => {
  nprogress.reset(); // Start the progress bar when the component mounts
  return () => {
    nprogress.complete(); // Complete the progress bar when the component unmounts
  };

}, [pathname]);
  return (
    <>
     <NavigationProgress color="red"/>
      <div className={classes.navbarMain}>
        {data.map((item) => (
          <Link href={item.link} key={item.label} 
          onClick={() => {         
            setActive(item.label);
            nprogress.set(50); 
        
          }}
          >
            <div className={`${classes.link} ${pathname === item.link ? classes.active : ''}`}   data-active={item.label === active ? true : undefined} >
                
              <item.icon className={classes.linkIcon} stroke={1.5} />
              <span>{item.label}</span>
           
            </div>
          </Link>
        ))}
      </div>

      <div className={classes.footer}>
        <Link href="/dashboard/settings"  className={classes.link}>
          <IconSwitchHorizontal className={classes.linkIcon} stroke={1.5} />
          <span>Change account</span>
        </Link>

        <a
          href="#"
          className={classes.link}
          onClick={(event) => event.preventDefault()}
        >
          <IconLogout className={classes.linkIcon} stroke={1.5} />
          <span>Logout</span>
        </a>
      </div>
    </>
  );
}
