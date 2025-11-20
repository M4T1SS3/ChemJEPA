import { HTMLAttributes, forwardRef } from 'react'

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  hover?: boolean
  noPadding?: boolean
}

const Card = forwardRef<HTMLDivElement, CardProps>(
  ({ className = '', children, hover = true, noPadding = false, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={`
          bg-elevated border border-border rounded-2xl shadow-md
          ${noPadding ? '' : 'p-8'}
          ${hover ? 'hover-lift cursor-pointer' : ''}
          transition-all duration-300 ease-out
          ${className}
        `}
        {...props}
      >
        {children}
      </div>
    )
  }
)

Card.displayName = 'Card'

export default Card
