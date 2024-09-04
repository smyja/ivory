import { useState, useEffect } from 'react';
import { Box, UnstyledButton, Collapse, Group, rem } from '@mantine/core';
import { usePathname, useRouter } from 'next/navigation';
import { NavigationProgress, nprogress } from '@mantine/nprogress';
import {
  IconBellRinging,
  IconFingerprint,
  IconSettings,
  Icon2fa,
  IconMessageChatbot,
  IconFiles,
  IconChevronRight,
  IconSwitchHorizontal,
  IconLogout,
} from '@tabler/icons-react';
import Link from 'next/link';
import classes from './NavbarSimpleColored.module.css';

const data = [
  { link: '/dashboard', label: 'Home', icon: IconBellRinging },
  { link: '/dashboard/chat', label: 'Chat', icon: IconMessageChatbot },
  { link: '/dashboard/connectors', label: 'Connectors', icon: Icon2fa },
  { link: '/dashboard/documents', label: 'Documents', icon: IconFiles },
  {
    label: 'Datasets',
    icon: IconFiles,
    links: [
      { link: '/dashboard/datasets', label: 'Overview' },
      { link: '/dashboard/datasets/new', label: 'New Dataset' },
    ],
  },
  { link: '/dashboard/insight', label: 'Insight', icon: IconFingerprint },
  { link: '/dashboard/settings', label: 'Settings', icon: IconSettings },
];

export function NavbarSimpleColored() {
  const pathname = usePathname();
  const router = useRouter();

  const findLabelByPath = (path) => {
    const matchingItem = data.find((item) =>
      item.link ? item.link === path : item.links?.some((subLink) => subLink.link === path)
    );
    return matchingItem ? matchingItem.label : 'Home';
  };

  const [active, setActive] = useState(findLabelByPath(pathname));
  const [openGroup, setOpenGroup] = useState('');

  useEffect(() => {
    nprogress.reset();
    setActive(findLabelByPath(pathname));
    return () => {
      nprogress.complete();
    };
  }, [pathname]);

  const handleLinkClick = (item) => {
    setActive(item.label);
    router.push(item.link);
    nprogress.set(50);
  };

  return (
    <>
      <NavigationProgress color="red" />
      <div className={classes.navbarMain}>
        {data.map((item) => {
          if (item.links) {
            return (
              <Box key={item.label} className={classes.collapsibleLink}>
                <UnstyledButton
                  onClick={() => setOpenGroup((prev) => (prev === item.label ? '' : item.label))}
                  className={`${classes.control} ${
                    active === item.label || item.links.some((subLink) => pathname === subLink.link)
                      ? classes.active
                      : ''
                  }`}
                >
                  <Group justify="space-between" gap={108}>
                    <Box style={{ display: 'flex', alignItems: 'center' }}>
                      <item.icon className={classes.linkIcon} stroke={1.5} />
                      <span>{item.label}</span>
                    </Box>
                    <IconChevronRight
                      className={classes.chevron}
                      stroke={1.5}
                      style={{
                        transform: openGroup === item.label ? 'rotate(90deg)' : 'none',
                      }}
                    />
                  </Group>
                </UnstyledButton>

                <Collapse in={openGroup === item.label}>
                  <div className={classes.subLinks}>
                    {item.links.map((subLink) => (
                      <Link
                        href={subLink.link}
                        key={subLink.label}
                        onClick={() => handleLinkClick(subLink)}
                      >
                        <div
                          className={`${classes.link} ${
                            pathname === subLink.link ? classes.active : ''
                          }`}
                        >
                          {subLink.label}
                        </div>
                      </Link>
                    ))}
                  </div>
                </Collapse>
              </Box>
            );
          } else {
            return (
                  <Link href={item.link} key={item.label} onClick={() => handleLinkClick(item)}>
                    <div className={`${classes.link} ${active === item.label ? classes.active : ''}`}>
                      <item.icon className={classes.linkIcon} stroke={1.5} />
                      <span>{item.label}</span>
                    </div>
                  </Link>
            );
          }
        })}
      </div>

      <div className={classes.footer}>
        <Link href="/dashboard/settings" className={classes.link}>
          <IconSwitchHorizontal className={classes.linkIcon} stroke={1.5} />
          <span>Change account</span>
        </Link>

        <a href="#" className={classes.link} onClick={(event) => event.preventDefault()}>
          <IconLogout className={classes.linkIcon} stroke={1.5} />
          <span>Logout</span>
        </a>
      </div>
    </>
  );
}
